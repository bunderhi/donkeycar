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

from docopt import docopt
import numpy as np

import donkeycar as dk
from donkeycar.parts.datastore import TubHandler
from donkeycar.parts.controller import LocalWebController
from donkeycar.parts.actuator import PCA9685, PWMSteering, PWMThrottle
from donkeycar.parts.path import Path, PathPlot, PlotCircle, PImage
from donkeycar.parts.realsense2 import RS_T265
from donkeycar.utils import *


def drive(cfg):
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
          inputs=['map/image'],
          outputs=['user/angle', 'user/throttle', 'user/mode', 'recording'],
          threaded=True)

    # we give the T265 no calib to indicated we don't have odom
    cfg.WHEEL_ODOM_CALIB = None

    #This dummy part to satisfy input needs of RS_T265 part.
    class NoOdom():
        def run(self):
            return 0.0

    V.add(NoOdom(), outputs=['enc/vel_m_s'])

    # This requires use of the Intel Realsense T265
    rs = RS_T265(image_output=False, calib_filename=cfg.WHEEL_ODOM_CALIB)
    V.add(rs, inputs=['enc/vel_m_s'], outputs=['rs/pos', 'rs/vel', 'rs/acc', 'rs/camera/left/img_array'], threaded=True)

    # Pull out the realsense T265 position stream, output 2d coordinates we can use to map.
    class PosStream:
        def run(self, pos):
            #y is up, x is right, z is backwards/forwards
            return pos.x, pos.z, pos.y
    V.add(PosStream(), inputs=['rs/pos'], outputs=['pos/x', 'pos/y', 'pos/z'])
    
    # Pull out the realsense T265 velocity stream.
    class VelStream:
        def run(self, vel):
            #y is up, x is right, z is backwards/forwards
            return vel.x, vel.z, vel.y
    V.add(VelStream(), inputs=['rs/vel'], outputs=['vel/x', 'vel/y', 'vel/z'])

    # This is the path object. It will record a path when distance changes and it travels
    # at least cfg.PATH_MIN_DIST meters. Except when we are in follow mode, see below...
    path = Path(min_dist=cfg.PATH_MIN_DIST)
    V.add(path, inputs=['pos/x', 'pos/y'], outputs=['path'], run_condition='run_user')

    # Here's an image we can map to.
    img = PImage(clear_each_frame=True)
    V.add(img, outputs=['map/image'])

    # This PathPlot will draw path on the image
    plot = PathPlot(scale=cfg.PATH_SCALE, offset=cfg.PATH_OFFSET)
    V.add(plot, inputs=['map/image', 'path'], outputs=['map/image'])

    # Plot a circle on the map where the car is located
    loc_plot = PlotCircle(scale=cfg.PATH_SCALE, offset=cfg.PATH_OFFSET)
    V.add(loc_plot, inputs=['map/image', 'pos/x', 'pos/y'], outputs=['map/image'])

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

    inputs=['map/image', 'user/angle', 'user/throttle', 'pos/x', 'pos/y', 'pos/z', 'vel/x', 'vel/y', 'vel/z']

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
    
    if args['drive']:      
        drive(cfg)