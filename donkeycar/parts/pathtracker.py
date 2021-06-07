'''
File: pathtracker.py
Path tracking simulation with Stanley steering control and PID speed control.

author: Atsushi Sakai (@Atsushi_twi)
Ref:
    - [Stanley: The robot that won the DARPA grand challenge](http://isl.ecst.csuchico.edu/DOCS/darpa2005/DARPA%202005%20Stanley.pdf)
    - [Autonomous Automobile Path Tracking](https://www.ri.cmu.edu/pub_files/2009/2/Automatic_Steering_Methods_for_Autonomous_Automobile_Path_Tracking.pdf)
'''
import numpy as np

class StanleyController(object):

    MIN_THROTTLE = 0.0
    MAX_THROTTLE = 1.0

    def __init__(self, cfg):
        self.cfg = cfg
        self.k = 0.5  # control gain
        self.Kp = cfg.KP  # speed proportional gain
        self.Kd = cfg.KD  # speed diferential gain
        self.Kta = 1.0 # accel to throttle ratio
        self.maxaccel = cfg.MAX_ACCEL
        self.L = 29  # [m] Wheel base of vehicle
        self.x = 0.
        self.y = 0.
        self.camx = 105.
        self.camy = 400.
        self.yaw = 0. # Current yaw (birdseye frame)
        self.throttle = 0. # current throttle setting
        self.img_count = 0
        self.timestamp = 0


    def constant_speed_control(self,v_target,v_current,throttle,dt):
        """
        Proportional control for the speed.
        :param target v: (float)
        :param current v: (float)
        :param previous v: (float)
        :param dt: (float) 
        :return target change in accel: (float)
        """
        v_correction = self.Kp * (v_target - v_current)
        accel_delta = v_correction / dt
        if accel_delta > self.maxaccel:
            accel_delta = self.maxaccel 
        if accel_delta < -self.maxaccel:
            accel_delta = -self.maxaccel 
        throttle = throttle + (accel_delta * self.Kta)  
        if throttle < self.MIN_THROTTLE: 
            throttle = self.MIN_THROTTLE
        if throttle > self.MAX_THROTTLE: 
            throttle = self.MAX_THROTTLE
        return throttle


    def stanley_control(self, cx, cy, cyaw, v, last_target_idx):
        """
        Stanley steering control.
        :param state: (State object)
        :param cx: ([float])
        :param cy: ([float])
        :param cyaw: ([float])
        :param last_target_idx: (int)
        :return: (float, int)
        """
        current_target_idx, error_front_axle = self.calc_target_index(cx, cy)

        if last_target_idx >= current_target_idx:
            current_target_idx = last_target_idx

        # theta_e corrects the heading error
        theta_e = self.normalize_angle(cyaw[current_target_idx] - self.yaw)
        # theta_d corrects the cross track error
        theta_d = np.arctan2(self.k * error_front_axle, v)
        # Steering control
        delta = theta_e + theta_d
        
        return delta, current_target_idx


    def normalize_angle(self,angle):
        """
        Normalize an angle to [-pi, pi].
        :param angle: (float)
        :return: (float) Angle in radian in [-pi, pi]
        """
        while angle > np.pi:
            angle -= 2.0 * np.pi

        while angle < -np.pi:
            angle += 2.0 * np.pi
        
        return angle


    def calc_target_index(self, cx, cy):
        """
        Compute index in the trajectory list of the target.
        :param state: (State object)
        :param cx: [float]
        :param cy: [float]
        :return: (int, float)
        """
        
        # Calc front axle position
        fx = self.camx + self.L * np.cos(self.yaw)
        fy = self.camy + self.L * np.sin(self.yaw)

        # Search nearest point index
        dx = [fx - icx for icx in cx]
        dy = [fy - icy for icy in cy]
        d = np.hypot(dx, dy)
        target_idx = np.argmin(d)

        # Project RMS error onto front axle vector
        front_axle_vec = [-np.cos(self.yaw + np.pi / 2),
                            -np.sin(self.yaw + np.pi / 2)]
        error_front_axle = np.dot([dx[target_idx], dy[target_idx]], front_axle_vec)

        return target_idx, error_front_axle

    def run(self,img_count,x,y,yaw,velturn,velfwd,rax,ray,ryaw,speedprofile,timestamp,runstate):
        if img_count > self.img_count:
            self.x = x
            self.y = y
            self.camx = 105
            self.camy = 400
            self.img_count = img_count  
        else:
            dx = (x - self.x) * 100
            dy = (y - self.y) * 100 
            self.camy = self.camy - (np.cos(yaw)*dy - np.sin(yaw)*dx)   # rotate velocity by yaw angle to the camera frame
            self.camx = self.camx + (np.sin(yaw)*dy + np.cos(yaw)*dx)
            self.x = x
            self.y = y
            print(f'reuse situation {self.camx},{self.camy}')
        
        v = np.abs(np.hypot(velfwd, velturn))
        self.yaw = np.arctan2(velfwd, velturn) - (np.pi / 2.)
        dt = (timestamp - self.timestamp) / 1000. # dt in seconds  
        
        if runstate == 'running':
            target_idx, _ = self.calc_target_index(rax, ray)
            target_speed = speedprofile[target_idx]
            delta, target_idx = self.stanley_control(rax, ray, ryaw, v, target_idx)
        else: # if the car is not in a running state keep it stopped
            target_speed = 0.0
            delta = -np.pi/2
        yaw_correction = np.arctan2(velfwd, velturn) - delta
        throttle = self.constant_speed_control(target_speed, v, self.throttle, dt)
        print(np.arctan2(velfwd, velturn),delta,yaw_correction, v, target_speed, throttle)
        self.throttle = throttle # for next time around
        self.timestamp = timestamp
        return self.camx,self.camy,yaw_correction,throttle
