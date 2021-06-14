'''
Author: Tawn Kramer
File: realsense2.py
Date: April 14 2019
Notes: Parts to input data from Intel Realsense 2 cameras
'''
import time
import logging

import numpy as np
import cv2
import os
from math import tan, pi, asin, atan2
import pyrealsense2 as rs

class RPY:
    def __init__(self, rotation):
        w = rotation.w
        x = -rotation.z
        y = rotation.x
        z = -rotation.y

        self.pitch =  -asin(2.0 * (x*z - w*y)) 
        self.roll  =  atan2(2.0 * (w*x + y*z), w*w - x*x - y*y + z*z) 
        self.yaw   =  atan2(2.0 * (w*z + x*y), w*w + x*x - y*y - z*z) # * 180.0 / pi (rad to degree)


class RS_T265(object):
    '''
    The Intel Realsense T265 camera is a device which uses an imu, twin fisheye cameras,
    and an Movidius chip to do sensor fusion and emit a world space coordinate frame that 
    is remarkably consistent.
    '''

    """
    Returns R, T transform from src to dst
    """
    def get_extrinsics(self,src,dst):
        extrinsics = src.get_extrinsics_to(dst)
        R = np.reshape(extrinsics.rotation, [3,3]).T
        T = np.array(extrinsics.translation)
        return (R, T)

    """
    Returns a camera matrix K from librealsense intrinsics
    """
    def camera_matrix(self,intrinsics):
        return np.array([[intrinsics.fx,             0, intrinsics.ppx],
                        [            0, intrinsics.fy, intrinsics.ppy],
                        [            0,             0,              1]])

    """
    Returns the fisheye distortion from librealsense intrinsics
    """
    def fisheye_distortion(self,intrinsics):
        return np.array(intrinsics.coeffs[:4])


    def __init__(self, image_output=False, calib_filename=None):
        # Using the image_output will grab two image streams from the fisheye cameras but return only one.
        # This can be a bit much for USB2, but you can try it. Docs recommend USB3 connection for this.
        self.image_output = image_output

        # When we have and encoder, this will be the last vel measured. 
        self.enc_vel_ms = 0.0
        self.wheel_odometer = None

        # Declare RealSense pipeline, encapsulating the actual device and sensors
        print("starting T265")
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.pose)
        # bug workaround
        #profile = cfg.resolve(self.pipe)
        #dev = profile.get_device()
        #tm2 = dev.as_tm2()
        

        if self.image_output:
            cfg.enable_stream(rs.stream.fisheye, 1) # Left camera
            cfg.enable_stream(rs.stream.fisheye, 2) # Right camera
            #disable wheel odometery for now due to bug
            #if calib_filename is not None:
            #    pose_sensor = tm2.first_pose_sensor()
            #    self.wheel_odometer = pose_sensor.as_wheel_odometer() 

                # calibration to list of uint8
            #    f = open(calib_filename)
            #    chars = []
            #    for line in f:
            #        for c in line:
            #            chars.append(ord(c))  # char to uint8

            # load/configure wheel odometer
            print("loading wheel config", calib_filename)
            #    self.wheel_odometer.load_wheel_odometery_config(chars)   


        # Start streaming with requested config
        self.pipe.start(cfg)
        self.running = True
        print("Warning: T265 needs a warmup period of a few seconds before it will emit tracking data.")
        if self.image_output:
            # Configure the OpenCV stereo algorithm. See
            # https://docs.opencv.org/3.4/d2/d85/classcv_1_1StereoSGBM.html for a
            # description of the parameters
            #window_size = 5
            min_disp = 0
            # must be divisible by 16
            num_disp = 112 - min_disp
            self.max_disp = min_disp + num_disp
            # Retreive the stream and intrinsic properties for both cameras
            profiles = self.pipe.get_active_profile()
            streams = {"left"  : profiles.get_stream(rs.stream.fisheye, 1).as_video_stream_profile(),
                        "right" : profiles.get_stream(rs.stream.fisheye, 2).as_video_stream_profile()}
            intrinsics = {"left"  : streams["left"].get_intrinsics(),
                            "right" : streams["right"].get_intrinsics()}

            # Print information about both cameras
            print("Left camera:",  intrinsics["left"])
            print("Right camera:", intrinsics["right"])

            # Translate the intrinsics from librealsense into OpenCV
            K_left  = self.camera_matrix(intrinsics["left"])
            D_left  = self.fisheye_distortion(intrinsics["left"])
            K_right = self.camera_matrix(intrinsics["right"])
            D_right = self.fisheye_distortion(intrinsics["right"])
            #(width, height) = (intrinsics["left"].width, intrinsics["left"].height)

            # Get the relative extrinsics between the left and right camera
            (R, T) = self.get_extrinsics(streams["left"], streams["right"])
            # We need to determine what focal length our undistorted images should have
            # in order to set up the camera matrices for initUndistortRectifyMap.  We
            # could use stereoRectify, but here we show how to derive these projection
            # matrices from the calibration and a desired height and field of view

            # We calculate the undistorted focal length:
            #
            #         h
            # -----------------
            #  \      |      /
            #    \    | f  /
            #     \   |   /
            #      \ fov /
            #        \|/
            stereo_fov_rad = 90 * (pi/180)  # 90 degree desired fov
            stereo_height_px = 300          # 300x300 pixel stereo output
            stereo_focal_px = stereo_height_px/2 / tan(stereo_fov_rad/2)

            # We set the left rotation to identity and the right rotation
            # the rotation between the cameras
            R_left = np.eye(3)
            R_right = R

            # The stereo algorithm needs max_disp extra pixels in order to produce valid
            # disparity on the desired output region. This changes the width, but the
            # center of projection should be on the center of the cropped image
            stereo_width_px = stereo_height_px + self.max_disp
            stereo_size = (stereo_width_px, stereo_height_px)
            stereo_cx = (stereo_height_px - 1)/2 + self.max_disp
            stereo_cy = (stereo_height_px - 1)/2

            # Construct the left and right projection matrices, the only difference is
            # that the right projection matrix should have a shift along the x axis of
            # baseline*focal_length
            P_left = np.array([[stereo_focal_px, 0, stereo_cx, 0],
                                [0, stereo_focal_px, stereo_cy, 0],
                                [0,               0,         1, 0]])
            P_right = P_left.copy()
            P_right[0][3] = T[0]*stereo_focal_px

            # Construct Q for use with cv2.reprojectImageTo3D. Subtract max_disp from x
            # since we will crop the disparity later
            Q = np.array([[1, 0,       0, -(stereo_cx - self.max_disp)],
                            [0, 1,       0, -stereo_cy],
                            [0, 0,       0, stereo_focal_px],
                            [0, 0, -1/T[0], 0]])

            # Create an undistortion map for the left and right camera which applies the
            # rectification and undoes the camera distortion. This only has to be done
            # once
            m1type = cv2.CV_32FC1
            (lm1, lm2) = cv2.fisheye.initUndistortRectifyMap(K_left, D_left, R_left, P_left, stereo_size, m1type)
            (rm1, rm2) = cv2.fisheye.initUndistortRectifyMap(K_right, D_right, R_right, P_right, stereo_size, m1type)
            self.undistort_rectify = {"left"  : (lm1, lm2),
                                "right" : (rm1, rm2)}
        zero_vec = (0.0, 0.0, 0.0)
        self.pos = zero_vec
        self.vel = zero_vec
        self.acc = zero_vec
        self.rpy = zero_vec
        self.img = None

    def poll(self):

        if self.wheel_odometer:
            wo_sensor_id = 0  # indexed from 0, match to order in calibration file
            frame_num = 0  # not used
            v = rs.vector()
            v.x = -1.0 * self.enc_vel_ms  # m/s
            #v.z = -1.0 * self.enc_vel_ms  # m/s
            self.wheel_odometer.send_wheel_odometry(wo_sensor_id, frame_num, v)

        try:
            frames = self.pipe.wait_for_frames()
            logging.info("Wait for frames complete")
        except Exception as e:
            logging.error(e)
            return

        # Fetch pose frame
        pose = frames.get_pose_frame()
        logging.info("Fetch pose")
        if pose:
            data = pose.get_pose_data()
            self.pos = data.translation
            self.vel = data.velocity
            self.acc = data.acceleration
            self.rotation = data.rotation
            logging.info('realsense pos(%f, %f, %f)' % (self.pos.x, self.pos.y, self.pos.z))

            # Compute roll, pitch, and yaw
            self.rpy = RPY(self.rotation)
            
            logging.info('realsense RPandY(%f, %f, %f)' % (self.rpy.roll,self.rpy.pitch,self.rpy.yaw))
        
        if self.image_output:
            #We will just get one image for now.
            # Left fisheye camera frame
            left = frames.get_fisheye_frame(1)
            left_data = np.asanyarray(left.get_data())
            left_undistorted = cv2.remap(src = left_data,
                                       map1 = self.undistort_rectify["left"][0],
                                       map2 = self.undistort_rectify["left"][1],
                                       interpolation = cv2.INTER_LINEAR)
            self.img = cv2.cvtColor(left_undistorted[:,self.max_disp:], cv2.COLOR_GRAY2RGB)
            logging.info("Get image")

    def update(self):
        while self.running:
            self.poll()

    def run_threaded(self, enc_vel_ms):
        self.enc_vel_ms = enc_vel_ms
        return self.pos, self.vel, self.acc, self.rpy, self.img

    def run(self, enc_vel_ms):
        self.enc_vel_ms = enc_vel_ms
        self.poll()
        return self.run_threaded()

    def shutdown(self):
        self.running = False
        time.sleep(0.1)
        self.pipe.stop()


class RS_T265RAW(object):
    '''
    The Intel Realsense T265 camera is a device which uses an imu, twin fisheye cameras,
    and an Movidius chip to do sensor fusion and emit a world space coordinate frame.
    '''

    def __init__(self, image_output=False, calib_filename=None):
        # Using the image_output will grab two image streams from the fisheye cameras but return only one.
        # This can be a bit much for USB2, but you can try it. Docs recommend USB3 connection for this.
        self.image_output = image_output

        # When we have and encoder, this will be the last vel measured. 
        self.enc_vel_ms = 0.0
        self.wheel_odometer = None

        # Declare RealSense pipeline, encapsulating the actual device and sensors
        print("starting T265")
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.pose)
        # bug workaround
        #profile = cfg.resolve(self.pipe)
        #dev = profile.get_device()
        #tm2 = dev.as_tm2()
        

        if self.image_output:
            cfg.enable_stream(rs.stream.fisheye, 1) # Left camera
            cfg.enable_stream(rs.stream.fisheye, 2) # Right camera
            #disable wheel odometery for now due to bug
            #if calib_filename is not None:
            #    pose_sensor = tm2.first_pose_sensor()
            #    self.wheel_odometer = pose_sensor.as_wheel_odometer() 

                # calibration to list of uint8
            #    f = open(calib_filename)
            #    chars = []
            #    for line in f:
            #        for c in line:
            #            chars.append(ord(c))  # char to uint8

            # load/configure wheel odometer
            print("loading wheel config", calib_filename)
            #    self.wheel_odometer.load_wheel_odometery_config(chars)   


        # Start streaming with requested config
        self.pipe.start(cfg)
        self.running = True
        print("Warning: T265 needs a warmup period of a few seconds before it will emit tracking data.")
        if self.image_output:
            # Retreive the stream and intrinsic properties for both cameras
            profiles = self.pipe.get_active_profile()
            streams = {"left"  : profiles.get_stream(rs.stream.fisheye, 1).as_video_stream_profile(),
                        "right" : profiles.get_stream(rs.stream.fisheye, 2).as_video_stream_profile()}

        zero_vec = (0.0, 0.0, 0.0)
        self.pos = zero_vec
        self.vel = zero_vec
        self.acc = zero_vec
        self.rpy = zero_vec
        self.img = None

    def poll(self):

        if self.wheel_odometer:
            wo_sensor_id = 0  # indexed from 0, match to order in calibration file
            frame_num = 0  # not used
            v = rs.vector()
            v.x = -1.0 * self.enc_vel_ms  # m/s
            #v.z = -1.0 * self.enc_vel_ms  # m/s
            self.wheel_odometer.send_wheel_odometry(wo_sensor_id, frame_num, v)

        try:
            frames = self.pipe.wait_for_frames()
            logging.info("Wait for frames complete")
        except Exception as e:
            logging.error(e)
            return

        # Fetch pose frame
        pose = frames.get_pose_frame()
        logging.info("Fetch pose")
        if pose:
            data = pose.get_pose_data()
            self.pos = data.translation
            self.vel = data.velocity
            self.acc = data.acceleration
            self.rotation = data.rotation
            self.mapper_confidence = data.mapper_confidence
            self.timestamp = pose.get_timestamp()
            logging.info('realsense pos(%f, %f, %f)' % (self.pos.x, self.pos.y, self.pos.z))

            # Compute roll, pitch, and yaw
            self.rpy = RPY(self.rotation)
            logging.info('realsense RPandY(%f, %f, %f)' % (self.rpy.roll,self.rpy.pitch,self.rpy.yaw))
        
        if self.image_output:
            #We will just get one image for now.
            # Left fisheye camera frame
            left = frames.get_fisheye_frame(1)
            left_data = np.asanyarray(left.get_data())
            self.img = cv2.cvtColor(left_data, cv2.COLOR_GRAY2RGB)
            logging.info("Get image")

    def update(self):
        while self.running:
            self.poll() 

    def run_threaded(self, enc_vel_ms):
        self.enc_vel_ms = enc_vel_ms
        return self.pos, self.vel, self.acc, self.rpy, self.img

    def run(self, enc_vel_ms):
        self.enc_vel_ms = enc_vel_ms
        self.poll()
        return self.run_threaded(enc_vel_ms)

    def shutdown(self):
        self.running = False
        time.sleep(0.1)
        self.pipe.stop()

class ImgPreProcess(object):
    '''
    preprocess camera image for inference.
    '''
    def __init__(self, cfg):
        self.cfg = cfg
        self.gray = None
        self.crop_img = None
        self.im2 = None
        self.image = None
        self.inf_inputs = None
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        self.framecount = 0

    def run(self, img_arr):
        self.gray = cv2.cvtColor(img_arr,cv2.COLOR_RGB2GRAY)
        self.crop_img = self.gray[230:550, 130:770]
        self.crop_img = self.clahe.apply(self.crop_img)
        self.im2 = cv2.resize(self.crop_img,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)
        self.image = cv2.cvtColor(self.im2,cv2.COLOR_GRAY2RGB)
        self.inf_inputs = self.image.transpose(2,0,1).reshape(1,3,160,320)
        self.framecount += 1
        print(f'framecount {self.framecount}')
        return self.image,np.array(self.inf_inputs, dtype=np.float32, order='C')/255,self.framecount

class ImgAlphaBlend(object):
    '''
    Combine camera image and inference mask for fpv viewer.
    '''
    def __init__(self, cfg):
        self.cfg = cfg 
        if cfg.ALPHA:
            self.alpha = cfg.ALPHA
        else:
            self.alpha = 0.5
        self.beta = (1.0 - self.alpha)
        self.fill = np.zeros((2,160,320),dtype=np.uint8)
        self.t = time.time()
        self.camcount = 0
        self.infcount = 0
        self.fps = 0
        self.ips = 0
        self.timer = cfg.TIMER

    def run(self, mask, img, camcount, infcount):
        red = (mask*255).reshape(1,160,320)
        redmask = np.vstack((self.fill,red)).transpose(1,2,0)
        dst = cv2.addWeighted(redmask, self.alpha, img, self.beta, 0.0)
        if (self.timer and camcount % 100 == 0 and camcount != 0):
            e = time.time()
            inferences = infcount - self.infcount
            self.fps = round(100.0 / (e - self.t))
            self.ips = round(inferences / (e - self.t))
            print(f'fps: {self.fps} ips: {self.ips}')
            self.infcount = infcount
            self.t = time.time()
        if (self.timer):
            text = f'{self.ips}'          
            (label_width, label_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            y = label_height + baseline + 2
            dst = cv2.putText(dst,text,(2,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0), 2, cv2.LINE_AA) 
        return dst

