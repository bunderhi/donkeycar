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

#Operating modes
AIPILOT= True # Operating mode Manual or AIPilot
RECORD = False  # log to disk
FPV = True # show camera fpv and AI plan view 

#PATHS
CAR_PATH = PACKAGE_PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.join(CAR_PATH, 'data')
MODEL_PATH = os.path.join(CAR_PATH, 'models/model10.onnx')
ENGINE_PATH = os.path.join(CAR_PATH, 'engine/model10.trt')
#PATH_MASK = '/media/C63B-4FCD/data/**/*.jpg'
READ_PATH = '/media/C63B-4FCD/data/tub'

# TensorRT
RUN_THREADED = True  # Run the tensorRT engine threaded

#VEHICLE
DRIVE_LOOP_HZ = 20      # the vehicle loop will pause if faster than this speed.
MAX_LOOPS = None        # the vehicle loop can abort after this many iterations, when given a positive integer.


#9865, over rides only if needed, ie. TX2..
PCA9685_I2C_ADDR = 0x40     #I2C address, use i2cdetect to validate this number
PCA9685_I2C_BUSNUM = 1   #None will auto detect, which is fine on the pi. But other platforms should specify the bus num.

#DRIVETRAIN
#These options specify which chasis and motor setup you are using. Most are using SERVO_ESC.
#DC_STEER_THROTTLE uses HBridge pwm to control one steering dc motor, and one drive wheel motor
#DC_TWO_WHEEL uses HBridge pwm to control two drive motors, one on the left, and one on the right.
#SERVO_HBRIDGE_PWM use ServoBlaster to output pwm control from the PiZero directly to control steering, and HBridge for a drive motor.

#STEERING
STEERING_CHANNEL = 0            #channel on the 9685 pwm board 0-15
STEERING_LEFT_PWM = 460         #pwm value for full left steering
STEERING_RIGHT_PWM = 290        #pwm value for full right steering

#THROTTLE
THROTTLE_CHANNEL = 1            #channel on the 9685 pwm board 0-15
THROTTLE_FORWARD_PWM = 500      #pwm value for max forward throttle
THROTTLE_STOPPED_PWM = 370      #pwm value for no movement
THROTTLE_REVERSE_PWM = 220      #pwm value for max reverse throttle

#Camera
IMAGE_W = 224
IMAGE_H = 224
IMAGE_DEPTH = 3         # default RGB=3, make 1 for mono
CAMERA_FRAMERATE = DRIVE_LOOP_HZ
CAMERA_VFLIP = False
CAMERA_HFLIP = False
# For CSIC camera - If the camera is mounted in a rotated position, changing the below parameter will correct the output frame orientation
CSIC_CAM_GSTREAMER_FLIP_PARM = 0 # (0 => none , 4 => Flip horizontally, 6 => Flip vertically)
# Region of interst cropping
# only supported in Categorical and Linear models.
# If these crops values are too large, they will cause the stride values to become negative and the model with not be valid.
ROI_CROP_TOP = 0                    #the number of rows of pixels to ignore on the top of the image
ROI_CROP_BOTTOM = 0                 #the number of rows of pixels to ignore on the bottom of the image

ALPHA = 0.5  # Alpha blend value used for fpv image (camera image + mask)
TIMER = False  # Display FPS timing on fpv view

TARGET_SPEED = 150.0 / 100. # Target normal speed (m/s)
PATH_INCREMENT = 40  # The path planning will create path points every 40 
MAX_ACCEL = 1.0 # m/s**2
KP = 0.5  # Throttle PID proportional gain
KD = 0.5  # Throttle PID differential gain

#Odometry
HAVE_ODOM = False                   # Do you have an odometer? Uses pigpio 

#Intel T265
WHEEL_ODOM_CALIB = "calibration_odometry.json"
