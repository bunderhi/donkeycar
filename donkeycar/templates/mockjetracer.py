#!/usr/bin/env python3
"""
Scripts to drive a donkey 2 car with Realsense T265 

Usage:
    manage.py (drive) [--model=<model>] [--type=(linear|categorical|rnn|imu|behavior|3d|localizer|latent)]
    manage.py (train) [--tub=<tub1,tub2,..tubn>] [--file=<file> ...] (--model=<model>) [--transfer=<model>] [--type=(linear|categorical|rnn|imu|behavior|3d|localizer)] [--continuous] [--aug]


Options:
    -h --help          Show this screen.    
    -f --file=<file>   A text file containing paths to tub files, one per line. Option may be used more than once.
"""
import os
import time
import logging

from docopt import docopt
import numpy as np

import donkeycar as dk
from donkeycar.parts.datastore import TubHandler,TubReader
from donkeycar.parts.camera import ImageListCamera
from donkeycar.parts.controller import WebFpv
from donkeycar.parts.realsenseT265 import ImgPreProcess,ImgAlphaBlend,BirdseyeView
from donkeycar.parts.trt import TensorRTSegment

from donkeycar.utils import *


def drive(cfg,verbose=True):
    '''
    Construct a working robotic vehicle from many parts.
    Each part runs as a job in the Vehicle loop, calling either
    it's run or run_threaded method depending on the constructor flag `threaded`.
    All parts are updated one after another at the framerate given in
    cfg.DRIVE_LOOP_HZ assuming each part finishes processing in a timely manner.
    Parts may have named outputs and inputs. The framework handles passing named outputs
    to parts requesting the same named input.
    '''

    
    #Initialize car
    V = dk.vehicle.Vehicle()

    class HardcodeUserMode:
        def run(self):
            assert cfg.USERMODE is not None and cfg.AIPILOT is not None and cfg.RECORD is not None,'Missing config settings (USERMODE,AIPILOT,RECORD'
            return cfg.USERMODE,cfg.RECORD,cfg.AIPILOT
    V.add(HardcodeUserMode(), outputs=['user/mode','recording','AI/pilot'])
    

    # FPS Camera image viewer
    V.add(WebFpv(port=8890), inputs=['cam/fpv'], threaded=True)
    V.add(WebFpv(port=8891), inputs=['inf/RealMask'], threaded=True)

    # Mock camera from existing tub 
    inputs=['cam/image_array', 'pos/x', 'pos/y', 'pos/z', 'vel/x', 'vel/y', 'vel/z', 'rpy/roll', 'rpy/pitch', 'rpy/yaw']
    types=['image_array', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float']
    
    reader=TubReader(path=cfg.READ_PATH)
    V.add(reader,outputs=['input/record'])

    class ReadStream:
        def run(self, record_dict):
            print (len(record_dict))
            dict = record_dict[0]
            if record_dict is not None:
                img_array = dict['cam/image1']
                posx = dict['pos/x']
                posy = dict['pos/y']
                posz = dict['pos/z']
                velx = dict['vel/x']
                vely = dict['vel/y']
                velz = dict['vel/z']
                roll = dict['rpy/roll']
                pitch = dict['rpy/pitch']
                yaw = dict['rpy/yaw']
                return img_array,posx,posy,posz,velx,vely,velz,roll,pitch,yaw 
            return None,None,None,None,None,None,None,None,None,None
    
    V.add(ReadStream(),inputs=['input/record'],outputs=['cam/image_array','pos/x', 'pos/y', 'pos/z', 'vel/x', 'vel/y', 'vel/z', 'rpy/roll', 'rpy/pitch', 'rpy/yaw'])
    
    # Mock camera feed
    #cam = ImageListCamera(path_mask=cfg.PATH_MASK)
    #V.add(cam, outputs=['cam/image_array'], threaded=True)
    
    V.add(ImgPreProcess(cfg),
        inputs=['cam/image_array'],
        outputs=['cam/raw','cam/inf_input','cam/framecount']
        )

    # Create and load Freespace segmentation model
    trt = TensorRTSegment(cfg=cfg)

    start = time.time()
    print('loading model')
    trt.load(onnx_file_path=cfg.MODEL_PATH,engine_file_path=cfg.ENGINE_PATH)
    print('finished loading in %s sec.' % (str(time.time() - start)))

    V.add(trt, inputs=['cam/inf_input'],
        outputs=['inf/mask','inf/framecount'], run_condition='AI/pilot'
        )

    V.add(ImgAlphaBlend(cfg),
        inputs=['inf/mask','cam/raw','cam/framecount','inf/framecount'],
        outputs=['cam/fpv'], run_condition='AI/pilot'
        )

    V.add(BirdseyeView(cfg),
        inputs=['inf/mask'],
        outputs=['inf/RealMask'], run_condition='AI/pilot'
        )
    
    
    #add tub to save data
    inputs=['cam/fpv','inf/RealMask']
    types=['image_array','image_array']


    th = TubHandler(path=cfg.DATA_PATH)
    tub = th.new_tub_writer(inputs=inputs, types=types)
    V.add(tub, inputs=inputs, outputs=["tub/num_records"], run_condition='recording')

    #run the vehicle
    V.start(rate_hz=cfg.DRIVE_LOOP_HZ, 
            max_loop_count=cfg.MAX_LOOPS)


if __name__ == '__main__':

    args = docopt(__doc__)
    cfg = dk.load_config()

    logging.basicConfig(level=30)
    
    if args['drive']:      
        drive(cfg)