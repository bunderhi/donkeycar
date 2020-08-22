"""
CAR CONFIG

This file is read by your car application's manage.py script to change the car
performance.

EXAMPLE
-----------
import dk
cfg = dk.load_config(config_path='~/mycar/config.py')
print(cfg.CAMERA_RESOLUTION)

"""


import os

#PATHS
CAR_PATH = PACKAGE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(CAR_PATH, 'data')

#VEHICLE
DRIVE_LOOP_HZ = 20      # the vehicle loop will pause if faster than this speed.
MAX_LOOPS = None        # the vehicle loop can abort after this many iterations, when given a positive integer.


#9865, over rides only if needed, ie. TX2..
PCA9685_I2C_ADDR = 0x40     #I2C address, use i2cdetect to validate this number
PCA9685_I2C_BUSNUM = None   #None will auto detect, which is fine on the pi. But other platforms should specify the bus num.

#DRIVETRAIN
#These options specify which chasis and motor setup you are using. Most are using SERVO_ESC.
#DC_STEER_THROTTLE uses HBridge pwm to control one steering dc motor, and one drive wheel motor
#DC_TWO_WHEEL uses HBridge pwm to control two drive motors, one on the left, and one on the right.
#SERVO_HBRIDGE_PWM use ServoBlaster to output pwm control from the PiZero directly to control steering, and HBridge for a drive motor.

#STEERING
STEERING_CHANNEL = 1            #channel on the 9685 pwm board 0-15
STEERING_LEFT_PWM = 460         #pwm value for full left steering
STEERING_RIGHT_PWM = 290        #pwm value for full right steering

#THROTTLE
THROTTLE_CHANNEL = 0            #channel on the 9685 pwm board 0-15
THROTTLE_FORWARD_PWM = 500      #pwm value for max forward throttle
THROTTLE_STOPPED_PWM = 370      #pwm value for no movement
THROTTLE_REVERSE_PWM = 220      #pwm value for max reverse throttle



#Path following
PATH_FILENAME = "donkey_path.pkl"   # the path will be saved to this filename
PATH_SCALE = 10.0                   # the path display will be scaled by this factor in the web page
PATH_OFFSET = (255, 255)            # 255, 255 is the center of the map. This offset controls where the origin is displayed.
PATH_MIN_DIST = 0.3                 # after travelling this distance (m), save a path point
PID_P = -3.0                        # proportional mult for PID path follower
PID_I = 0.000                       # integral mult for PID path follower
PID_D = -0.2                        # differential mult for PID path follower
PID_THROTTLE = 0.25                 # constant throttle value during path following
SAVE_PATH_BTN = "cross"             # joystick button to save path
RESET_ORIGIN_BTN = "triangle"       # joystick button to press to move car back to origin

#Odometry
HAVE_ODOM = False                   # Do you have an odometer? Uses pigpio 

#Intel T265
WHEEL_ODOM_CALIB = "calibration_odometry.json"
