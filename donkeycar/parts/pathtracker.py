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

    def __init__(self, cfg):
        self.k = 0.5  # control gain
        self.Kp = 0.9  # speed proportional gain
        self.L = 29  # [m] Wheel base of vehicle
        self.x = 105.
        self.y = 400.
        self.v  # current Velocity
        self.yaw # Current yaw (birdseye frame)
        self.target_speed = cfg.TARGET_SPEED # target velocity in cm/s 


    def pid_control(self,target,current):
        """
        Proportional control for the speed.
        :param target: (float)
        :param current: (float)
        :return: (float)
        """
        return self.Kp * (target - current)


    def stanley_control(self, cx, cy, cyaw, last_target_idx):
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
        theta_d = np.arctan2(self.k * error_front_axle, self.v)
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
        fx = self.x + self.L * np.cos(self.yaw)
        fy = self.y + self.L * np.sin(self.yaw)

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

    def run(self,velturn,velfwd,rax,ray,ryaw):
        target_idx, _ = self.calc_target_index(self, rax, ray)
        self.v = np.hypot(velfwd, velturn)
        self.yaw = np.arctan2(velfwd, velturn) - (np.pi / 2.)
        accel = self.pid_control(self.target_speed, self.v)
        delta, target_idx = self.stanley_control(rax, ray, ryaw, target_idx)
        return delta,accel
