import sys,os

import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as tensorrt
from pathlib import Path


EXPLICIT_BATCH = 1 << (int)(tensorrt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

def GiB(val):
    return val * 1 << 30

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class TensorRTSegment(object):
    '''
    Use TensorRT to perform a image freespace segmentation inference.
    '''
    
    def __init__(self, cfg, *args, **kwargs):
        self.logger = tensorrt.Logger()
        self.cfg = cfg
        self.engine = None
        self.inputs = None
        self.outputs = None
        self.bindings = None
        self.stream = None
        self.infcount = 0
    
    def compile(self):
        print('Nothing to compile')
    
    def load(self, onnx_file_path, engine_file_path):
        print('Building CUDA Engine')
        self.engine = TensorRTSegment.get_engine(onnx_file_path=onnx_file_path,engine_file_path=engine_file_path) 
        print('Allocating Buffers')
        self.inputs, self.outputs, self.bindings, self.stream = TensorRTSegment.allocate_buffers(self.engine)
        print('Ready')

    def run(self, inf_inputs):
        # Do inference
        print('Running inference on image')
        trt_outputs = []
        # Set host input to the image. The do_inference_v2 function will copy the input to the GPU before executing.
        self.inputs[0].host = inf_inputs
        with self.engine.create_execution_context() as context:
            trt_outputs = TensorRTSegment.do_inference_v2(context=context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)
        mask = (trt_outputs[0] > 0.4).astype(np.uint8).reshape(160,320)
        self.infcount += 1
        return mask,self.infcount

    # Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
    @classmethod
    def allocate_buffers(cls,engine):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        for binding in engine:
            size = tensorrt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = tensorrt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream
    
    @classmethod
    def do_inference_v2(self,context, bindings, inputs, outputs, stream):
        """This function is generalized for multiple inputs/outputs for full dimension networks."""
        # inputs and outputs are expected to be lists of HostDeviceMem objects.
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # Run inference.
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        # Synchronize the stream
        stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in outputs]

    @classmethod
    def get_engine(self,onnx_file_path, engine_file_path=""):
        """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
        def build_engine(self):
            """Takes an ONNX file and creates a TensorRT engine to run inference with"""
            with tensorrt.Builder(tensorrt.ILogger) as builder, builder.create_network(EXPLICIT_BATCH) as network, tensorrt.OnnxParser(network, self.logger()) as parser:
                builder.max_workspace_size = 1 << 28 # 256MiB
                builder.max_batch_size = 1
                builder.fp16_mode = True
                # Parse model file
                if not os.path.exists(onnx_file_path):
                    print('ONNX file {} not found'.format(onnx_file_path))
                    exit(0)
                print('Loading ONNX file from path {}...'.format(onnx_file_path))
                with open(onnx_file_path, 'rb') as model:
                    print('Beginning ONNX file parsing')
                    if not parser.parse(model.read()):
                        print ('ERROR: Failed to parse the ONNX file.')
                        for error in range(parser.num_errors):
                            print (parser.get_error(error))
                        return None
                #  Reshape input to batch size 1
                network.get_input(0).shape = [1, 3, 160, 320]
                print('Completed parsing of ONNX file')
                print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
                engine = builder.build_cuda_engine(network)
                print("Completed creating Engine")
                with open(engine_file_path, "wb") as f:
                    f.write(engine.serialize())
                return engine
    
        if os.path.exists(engine_file_path):
            # If a serialized engine exists, use it instead of building an engine.
            print("Reading engine from file {}".format(engine_file_path))
            with open(engine_file_path, "rb") as f, tensorrt.Runtime(tensorrt.ILogger) as runtime:
                return runtime.deserialize_cuda_engine(f.read())
        else:
            return build_engine(self)