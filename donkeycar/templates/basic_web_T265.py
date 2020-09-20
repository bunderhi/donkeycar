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
from donkeycar.parts.datastore import TubHandler
from donkeycar.parts.camera import CSICamera
from donkeycar.parts.controller import LocalWebController
from donkeycar.parts.actuator import PCA9685, PWMSteering, PWMThrottle
from donkeycar.parts.realsenseT265 import RS_T265
from donkeycar.parts.nextion_controller import NextionController

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
 
    V.add(LocalWebController(), 
          inputs=['cam/image1'],
          outputs=['user/angle', 'user/throttle', 'user/mode', 'recording'],
          threaded=True)

    # we give the T265 no calib to indicated we don't have odom
    cfg.WHEEL_ODOM_CALIB = None

    #This dummy part to satisfy input needs of RS_T265 part.
    class NoOdom():
        def run(self):
            return 0.0

    # Wide Angle Camera 
    #cam = CSICamera(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH, framerate=cfg.CAMERA_FRAMERATE, gstreamer_flip=cfg.CSIC_CAM_GSTREAMER_FLIP_PARM)
    #V.add(cam, inputs=[], outputs=['cam/image2'], threaded=True)

    V.add(NoOdom(), outputs=['enc/vel_m_s'])

    #This part implements a system console display
    # For now it only controls record on/off 
    console = NextionController()
    V.add(console(), outputs=['recording'])  

    # This requires use of the Intel Realsense T265
    rs = RS_T265(image_output=True, calib_filename=cfg.WHEEL_ODOM_CALIB)
    V.add(rs, inputs=['enc/vel_m_s'], outputs=['rs/pos', 'rs/vel', 'rs/acc', 'rs/rpy', 'cam/image1'], threaded=True)

    # Pull out the realsense T265 position stream, output 2d coordinates we can use to map.
    class PosStream:
        def run(self, pos):
            #y is up, x is right, z is backwards/forwards
            zero_vec = (0.0, 0.0, 0.0)
            try:
                return pos.x, pos.z, pos.y
            except:
                return zero_vec
    V.add(PosStream(), inputs=['rs/pos'], outputs=['pos/x', 'pos/z', 'pos/y'])
    
    # Pull out the realsense T265 rotation stream.
    class RPYStream:
        def run(self, rpy):
            #rpy - roll, pitch yaw
            zero_vec = (0.0, 0.0, 0.0)
            try:
                return rpy.roll, rpy.pitch, rpy.yaw
            except:
                return zero_vec
    V.add(RPYStream(), inputs=['rs/rpy'], outputs=['rpy/roll', 'rpy/pitch', 'rpy/yaw'])


    #Drive train setup
    steering_controller = PCA9685(cfg.STEERING_CHANNEL, cfg.PCA9685_I2C_ADDR, busnum=cfg.PCA9685_I2C_BUSNUM)
    steering = PWMSteering(controller=steering_controller,
                                    left_pulse=cfg.STEERING_LEFT_PWM, 
                                    right_pulse=cfg.STEERING_RIGHT_PWM)
    
    throttle_controller = PCA9685(cfg.THROTTLE_CHANNEL, cfg.PCA9685_I2C_ADDR, busnum=cfg.PCA9685_I2C_BUSNUM)
    throttle = PWMThrottle(controller=throttle_controller,
                                    max_pulse=cfg.THROTTLE_FORWARD_PWM,
                                    zero_pulse=cfg.THROTTLE_STOPPED_PWM, 
                                    min_pulse=cfg.THROTTLE_REVERSE_PWM)

    V.add(steering, inputs=['user/angle'])
    V.add(throttle, inputs=['user/throttle'])
    
    #add tub to save data

    inputs=['cam/image1', 'user/angle', 'user/throttle', 'pos/x', 'pos/y', 'pos/z', 'rpy/roll', 'rpy/pitch', 'rpy/yaw']

    types=['image_array', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float']


    th = TubHandler(path=cfg.DATA_PATH)
    tub = th.new_tub_writer(inputs=inputs, types=types)
    V.add(tub, inputs=inputs, outputs=["tub/num_records"], run_condition='recording')

    print("You can now go to <your hostname.local>:8887 to drive your car.")

    #run the vehicle
    V.start(rate_hz=cfg.DRIVE_LOOP_HZ, 
            max_loop_count=cfg.MAX_LOOPS)


if __name__ == '__main__':

    args = docopt(__doc__)
    cfg = dk.load_config()

    logging.basicConfig(level=30)
    
    if args['drive']:      
        drive(cfg)