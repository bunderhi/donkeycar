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
import math

import donkeycar as dk
from donkeycar.parts.datastore import TubHandler
from donkeycar.parts.web_controller.web import WebFpv, WebConsole
from donkeycar.parts.realsenseT265 import RS_T265RAW, ImgPreProcess, ImgAlphaBlend
from donkeycar.parts.planner import BirdseyeView, PlanPath, PlanMap
from donkeycar.parts.pathtracker import StanleyController
from donkeycar.parts.trt import TensorRTSegment
from donkeycar.parts.actuator import PCA9685, PWMSteering, PWMThrottle

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
    
    # Vehicle control web console
    #V.add(WebConsole(cfg),inputs=['RUN/State'],outputs=['RUN/State'],threaded=True)
    
    # Vehicle control web console is not currently reliable so hardcode run state
    class HardcodeState():
        def __init__(self):
            self.starttime = time.time()
        
        def run(self,runstate):
            if runstate == 'running':
                return runstate
            currenttime = time.time()
            if currenttime - self.starttime > 5.0:
                return 'running'
            else:
                return 'ready'


    V.add(HardcodeState(), inputs=['RUN/State'],outputs=['RUN/State'])
    
    
    # FPS Camera image viewer
    #if cfg.FPV:
    #    V.add(WebFpv(port=8890), inputs=['cam/fpv'], threaded=True,run_condition='fpv')
    #if cfg.FPV and cfg.AIPILOT:
    #    V.add(WebFpv(port=8891), inputs=['plan/map'], threaded=True,run_condition='AI/fpv')
    
    # we give the T265 no calib to indicated we don't have odom
    cfg.WHEEL_ODOM_CALIB = None

    #This dummy part to satisfy input needs of RS_T265 part.
    class NoOdom():
        def run(self):
            return 0.0

    V.add(NoOdom(), outputs=['enc/vel_m_s'])

    # This requires use of the Intel Realsense T265
    rs = RS_T265RAW(image_output=True, calib_filename=cfg.WHEEL_ODOM_CALIB)
    V.add(rs, inputs=['enc/vel_m_s'], outputs=['rs/pos', 'rs/vel', 'rs/acc', 'rs/rpy', 'cam/image_array'], threaded=True)

    class ReadStream:
        def run(self, pos,vel,rpy):
            try:
                posx = pos.x   # real posx = camera posx
                posz = -pos.y  # real posz = camera -posy
                posy = -pos.z   # real posy = camera -posz
                xvel = vel.x   # left/right vel  
                #vely = record[5]   
                yvel = vel.z   # forward vel  
                roll = rpy.roll
                pitch = rpy.pitch
                yaw = rpy.yaw   # math.radians(record[9])   #yaw was in degrees originally
                fwdvel = math.cos(yaw)*yvel - math.sin(yaw)*xvel   # rotate velocity by yaw angle to the camera frame
                turnvel = math.sin(yaw)*yvel + math.cos(yaw)*xvel
                return posx,posy,posz,turnvel,fwdvel,roll,pitch,yaw
            except:
                zero_vec = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                return zero_vec
    
    V.add(ReadStream(),inputs=['rs/pos','rs/vel','rs/rpy'],
                    outputs=['pos/x', 'pos/y', 'pos/z', 'vel/turn','vel/fwd', 'rpy/roll', 'rpy/pitch', 'rpy/yaw'])
    
    V.add(ImgPreProcess(cfg),
        inputs=['cam/image_array'],
        outputs=['cam/raw','cam/inf_input','cam/framecount']
        )

    class AIWarmup:
            '''
            Start part processing based on AI warmup state
            '''
            def __init__(self, cfg):
                self.cfg = cfg
 
            def run(self,inf_input,mask,runstate):
                if inf_input is None:   # camera not ready
                    return False,False,False,False,False,False,'initializing'
                if self.cfg.AIPILOT== False:   # manual mode
                    return False,False,self.cfg.RECORD,True,False,False,'ready'
                if mask is None:  # inference not ready
                    return True,False,False,False,False,False,'initializing'
                if runstate == 'running':  # vehicle running 
                    return True,True,self.cfg.RECORD,True,True,True,runstate
                else: # vehicle ready waiting for start cmd
                    return True,True,self.cfg.RECORD,True,True,True,'ready'
                


    V.add(AIWarmup(cfg), inputs=['cam/inf_input','inf/mask','RUN/State'],
             outputs=['AI/pilot','AI/processing','recording','fpv','AI/fpv','AI/running','RUN/State'])
    
    if cfg.AIPILOT:
        V.add(TensorRTSegment(cfg), 
            inputs=['cam/inf_input','cam/framecount'],
            outputs=['inf/mask','inf/framecount'], run_condition='AI/pilot', threaded=cfg.RUN_THREADED
            )

        V.add(ImgAlphaBlend(cfg),
            inputs=['inf/mask','cam/raw','cam/framecount','inf/framecount'],
            outputs=['cam/fpv'], run_condition='AI/fpv'
            )

        V.add(BirdseyeView(cfg),
            inputs=['inf/mask','inf/framecount'],
            outputs=['plan/freespace'], run_condition='AI/processing'
            )
        
        V.add(PlanPath(cfg),
            inputs=['plan/freespace','inf/framecount'],
            outputs=['plan/waypointx','plan/waypointy','plan/pathx','plan/pathy','plan/pathyaw','plan/speedprofile'], run_condition='AI/processing'
            )
        
        V.add(StanleyController(cfg),
            inputs=['inf/framecount','pos/x','pos/y','rpy/yaw','vel/turn','vel/fwd','plan/pathx','plan/pathy','plan/pathyaw','plan/speedprofile','RUN/State'],
            outputs=['cam/x','cam/y','plan/delta','plan/throttle'], run_condition='AI/processing'
            )
    
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

        V.add(steering, inputs=['plan/delta'], outputs=['AI/steeringpulse'],run_condition='AI/running')
        V.add(throttle, inputs=['plan/throttle'], outputs=['AI/throttlepulse'],run_condition='AI/running')
        
        V.add(PlanMap(cfg),
            inputs=['plan/freespace','cam/x','cam/y','vel/turn','vel/fwd','plan/pathx','plan/pathy','plan/delta','plan/throttle','AI/steeringpulse','AI/throttlepulse'],
            outputs=['plan/map'], run_condition='AI/fpv'
            )
            
    if cfg.AIPILOT:
        #add tub to save AI pilot data
        tinputs=['cam/fpv','plan/map','pos/x','pos/y','pos/z','vel/turn','vel/fwd','rpy/yaw','plan/delta','plan/throttle','AI/steeringpulse','AI/throttlepulse']
        types=['image_array','image_array', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'float', 'int', 'int']
    else:
        #add tub to save manual pilot data
        tinputs=['cam/raw','pos/x','pos/y','pos/z','vel/turn','vel/fwd','rpy/yaw']
        types=['image_array', 'float', 'float', 'float', 'float', 'float', 'float']
    
    th = TubHandler(path=cfg.DATA_PATH)
    tub = th.new_tub_writer(inputs=tinputs, types=types)
    V.add(tub, inputs=tinputs, outputs=["tub/num_records"], run_condition='recording')


    #run the vehicle
    V.start(rate_hz=cfg.DRIVE_LOOP_HZ, 
            max_loop_count=cfg.MAX_LOOPS)


if __name__ == '__main__':

    args = docopt(__doc__)
    cfg = dk.load_config()

    logging.basicConfig(level=30)
    
    if args['drive']:      
        drive(cfg)