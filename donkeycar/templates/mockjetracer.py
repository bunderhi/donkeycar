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
from donkeycar.parts.camera import ImageListCamera
from donkeycar.parts.controller import WebFpv
from donkeycar.parts.realsenseT265 import ImgPreProcess, ImgAlphaBlend
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

    # FPS Camera image viewer
    V.add(WebFpv(), inputs=['cam/fpv'], threaded=True)

    # Mock camera feed
    cam = ImageListCamera(path_mask=cfg.PATH_MASK)
    V.add(cam, outputs=['cam/image_array'], threaded=True)
    
    V.add(ImgPreProcess(cfg),
        inputs=['cam/image_array'],
        outputs=['cam/raw','cam/inf_input','cam/framecount']
        )

    # Create and load Freespace segmentation model
    trt = TensorRTSegment(cfg=cfg)

    start = time.time()
    print('loading model')
    trt.load(onnx_file_path=cfg.onnx_file_path,engine_file_path=cfg.engine_file_path)
    print('finished loading in %s sec.' % (str(time.time() - start)))

    V.add(trt, inputs=['cam/inf_input'],
        outputs=['cam/mask','inf/framecount']
        )

    V.add(ImgAlphaBlend(cfg),
        inputs=['cam/mask','cam/raw','cam/framecount','inf/framecount'],
        outputs=['cam/fpv']
        )

    #run the vehicle
    V.start(rate_hz=cfg.DRIVE_LOOP_HZ, 
            max_loop_count=cfg.MAX_LOOPS)


if __name__ == '__main__':

    args = docopt(__doc__)
    cfg = dk.load_config()

    logging.basicConfig(level=30)
    
    if args['drive']:      
        drive(cfg)