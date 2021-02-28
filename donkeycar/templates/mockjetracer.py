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
from donkeycar.parts.realsenseT265 import ImgPreProcess,ImgAlphaBlend
from donkeycar.parts.planner import BirdseyeView, PlanPath, PlanMap
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
    V.add(WebFpv(port=8891), inputs=['plan/map'], threaded=True)

    # Mock camera from existing tub 
    inputs=['arg0', 'arg1', 'arg2', 'arg3', 'arg4', 'arg5', 'arg6', 'arg7', 'arg8', 'arg9']
    

    class InitializeReader:
        def run(self):
            return 'cam/image1','pos/x','pos/y','pos/z','vel/x','vel/y','vel/z','rpy/roll','rpy/pitch','rpy/yaw'
    V.add(InitializeReader(), outputs=inputs)


    reader=TubReader(path=cfg.READ_PATH)
    V.add(reader,inputs=inputs,outputs=['input/record'])

    class ReadStream:
        def run(self, record):
            print (len(record))
            if record is not None:
                img_array = record[0]
                posy = record[1]   # real posy = camera posx
                posz = -record[2]   # real posz = camera -posy
                posx = -record[3]   # real posx = camera -posz
                turnvel = record[4]   # left/right vel  = camera velx
                #vely = record[5]   
                fwdvel = record[6]   # forward vel  = camera -velz
                roll = record[7]
                pitch = record[8]
                yaw = record[9]
                return img_array,posx,posy,posz,turnvel,fwdvel,roll,pitch,yaw 
            return None,None,None,None,None,None,None,None,None
    
    V.add(ReadStream(),inputs=['input/record'],outputs=['cam/image_array', 'pos/x', 'pos/y', 'pos/z', 'vel/turn','vel/fwd', 'rpy/roll', 'rpy/pitch', 'rpy/yaw'])
    
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
        outputs=['plan/freespace'], run_condition='AI/pilot'
        )
    
    V.add(PlanPath(cfg),
        inputs=['plan/freespace'],
        outputs=['plan/waypointx','plan/waypointy','plan/pathx','plan/pathy'], run_condition='AI/pilot'
        )
    
    V.add(PlanMap(cfg),
        inputs=['plan/freespace','vel/turn','vel/fwd','plan/waypointx','plan/waypointy','plan/pathx','plan/pathy'],
        outputs=['plan/map'], run_condition='AI/pilot'
        )

    #add tub to save data
    inputs=['plan/map','pos/x','pos/y','pos/z','vel/turn','vel/fwd','rpy/roll','rpy/pitch','rpy/yaw']
    types=['image_array', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float']


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