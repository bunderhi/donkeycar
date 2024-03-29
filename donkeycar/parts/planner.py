'''
File: planner.py
Parts for path planning and tracking

includes Cubic spline planner
Author: Atsushi Sakai(@Atsushi_twi)
'''
import numpy as np
import cv2
import bisect
import math
import donkeycar as dk

class BirdseyeView(object):
    '''
    Produce a undistorted birdseye view for the inference mask.
    '''
    def __init__(self, cfg):
        self.cfg = cfg
        leftcam = {
        "width": 848,
        "height": 800, 
        "ppx": 432.782, "ppy": 406.656, 
        "fx": 285.247, "fy": 286.178, 
        "model": 5, 
        "coeffs": [-0.00460218, 0.0404374, -0.0388418, 0.00706689, 0]
        }
        # Translate the intrinsics from librealsense into OpenCV
        K  = self.camera_matrix(leftcam)
        D  = self.fisheye_distortion(leftcam)
        DIM = (leftcam["width"], leftcam["height"])
        print("camera_matrix:", K)
        print("distortion:",D)
        print("camera:", DIM)
        # create Undistort map for fisheye correction
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_32FC1) 
        # create Perspective Transform for birdseye view
        srcpts = np.float32([[371,455],[337,550],[472,472],[538,548]])  # mat + banister pts
        dstpts = np.float32([[27,65],[73,314],[133,194],[133,314]])
        self.M = cv2.getPerspectiveTransform(srcpts,dstpts)
        self.birdseye_mask = None
        self.img_count = 0

    def undistort(self,img):    
        """
        Perform a fisheye undistort  
        """
        undistorted_img = cv2.remap(img, self.map1, self.map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)    
        return undistorted_img 

    def reverse(self,img):
        """
        Reverse the preprocessing performed on an image 
        """
        img = cv2.cvtColor(img*255,cv2.COLOR_GRAY2RGB)
        original = np.zeros((800,848,3),dtype=np.uint8)
        img = cv2.resize(img,None,fx=2.0,fy=2.0,interpolation=cv2.INTER_AREA)
        original[230:550, 130:770] = img
        return original

    def warpperspective(self,img):
        """
        Create a birdseye view from image 
        """
        dst = cv2.warpPerspective(img,self.M,(200,400))
        return dst

    def camera_matrix(self,intrinsics):
        """
        Returns a camera matrix K from librealsense intrinsics
        """
        return np.array([[intrinsics["fx"],             0, intrinsics["ppx"]],
                        [            0, intrinsics["fy"], intrinsics["ppy"]],
                        [            0,             0,              1]])

    def fisheye_distortion(self,intrinsics):
        """
        Returns the fisheye distortion from librealsense intrinsics
        """
        return np.array(intrinsics["coeffs"][:4])


    def run(self,img,img_count):
        if img_count > self.img_count:  # Only perform transforms if its a fresh mask
            original = self.reverse(img)
            undistorted_img = self.undistort(original)
            redm = cv2.cvtColor(undistorted_img,cv2.COLOR_RGB2GRAY).reshape(800,848)
            self.birdseye_mask = self.warpperspective(redm)
        self.img_count = img_count
        return self.birdseye_mask

class Spline:
    """
    Cubic Spline class
    """

    def __init__(self, x, y):
        self.b, self.c, self.d, self.w = [], [], [], []

        self.x = x
        self.y = y

        self.nx = len(x)  # dimension of x
        h = np.diff(x)

        # calc coefficient c
        self.a = [iy for iy in y]

        # calc coefficient c
        A = self.__calc_A(h)
        B = self.__calc_B(h)
        self.c = np.linalg.solve(A, B)
        #  print(self.c1)

        # calc spline coefficient b and d
        for i in range(self.nx - 1):
            self.d.append((self.c[i + 1] - self.c[i]) / (3.0 * h[i]))
            tb = (self.a[i + 1] - self.a[i]) / h[i] - h[i] * \
                (self.c[i + 1] + 2.0 * self.c[i]) / 3.0
            self.b.append(tb)

    def calc(self, t):
        """
        Calc position
        if t is outside of the input x, return None
        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = self.a[i] + self.b[i] * dx + \
            self.c[i] * dx ** 2.0 + self.d[i] * dx ** 3.0

        return result

    def calcd(self, t):
        """
        Calc first derivative
        if t is outside of the input x, return None
        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx ** 2.0
        return result

    def calcdd(self, t):
        """
        Calc second derivative
        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = 2.0 * self.c[i] + 6.0 * self.d[i] * dx
        return result

    def __search_index(self, x):
        """
        search data segment index
        """
        return bisect.bisect(self.x, x) - 1

    def __calc_A(self, h):
        """
        calc matrix A for spline coefficient c
        """
        A = np.zeros((self.nx, self.nx))
        A[0, 0] = 1.0
        for i in range(self.nx - 1):
            if i != (self.nx - 2):
                A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
            A[i + 1, i] = h[i]
            A[i, i + 1] = h[i]

        A[0, 1] = 0.0
        A[self.nx - 1, self.nx - 2] = 0.0
        A[self.nx - 1, self.nx - 1] = 1.0
        #  print(A)
        return A

    def __calc_B(self, h):
        """
        calc matrix B for spline coefficient c
        """
        B = np.zeros(self.nx)
        for i in range(self.nx - 2):
            B[i + 1] = 3.0 * (self.a[i + 2] - self.a[i + 1]) / \
                h[i + 1] - 3.0 * (self.a[i + 1] - self.a[i]) / h[i]
        return B


class Spline2D:
    """
    2D Cubic Spline class
    """

    def __init__(self, x, y):
        self.s = self.__calc_s(x, y)
        self.sx = Spline(self.s, x)
        self.sy = Spline(self.s, y)

    def __calc_s(self, x, y):
        dx = np.diff(x)
        dy = np.diff(y)
        self.ds = np.hypot(dx, dy)
        s = [0]
        s.extend(np.cumsum(self.ds))
        return s

    def calc_position(self, s):
        """
        calc position
        """
        x = self.sx.calc(s)
        y = self.sy.calc(s)

        return x, y

    def calc_curvature(self, s):
        """
        calc curvature
        """
        dx = self.sx.calcd(s)
        ddx = self.sx.calcdd(s)
        dy = self.sy.calcd(s)
        ddy = self.sy.calcdd(s)
        k = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2)**(3 / 2))
        return k

    def calc_yaw(self, s):
        """
        calc yaw
        """
        dx = self.sx.calcd(s)
        dy = self.sy.calcd(s)
        yaw = math.atan2(dy, dx)
        return yaw

class PlanPath(object):
    """
    Determine a goal and a travel path 
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.waypntx = [100,100,100,100]
        self.waypnty = [400,300,200,100]
        self.rax = []
        self.ray = []
        self.ryaw = []
        self.ds = cfg.PATH_INCREMENT  # 400/40 = 10 path points 
        self.img_count = 0

    def calc_spline_course(self,x, y):
        sp = Spline2D(x, y)
        s = list(np.arange(0, sp.s[-1], self.ds))

        rx, ry, ryaw, rk = [], [], [], []
        for i_s in s:
            ix, iy = sp.calc_position(i_s)
            rx.append(ix)
            ry.append(iy)
            ryaw.append(sp.calc_yaw(i_s))
            rk.append(sp.calc_curvature(i_s))
        return rx, ry, ryaw

    def calc_speed_profile(self,ryaw):
        target_speed = self.cfg.TARGET_SPEED
        speed_profile = [target_speed] * len(ryaw)
        # speed down as you near the planning horizon (goal)
        for i in range(1,3):
            speed_profile[-i] = target_speed / (5 - i)
            if speed_profile[-i] <= 0.01:
                speed_profile[-i] = 0.01
        return speed_profile


    def setgoal(self,mask):
        """ 
            Find the coordinates for a path to the end goal using the freespace mask (birdseye view) 
        """   
        def first_nonzero(arr, axis, invalid_val=-1):
            """ List leftmost nonzero entry for each row in an array """
            mask = arr!=0
            return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

        def last_nonzero(arr, axis, invalid_val=-1):
            """ List rightmost nonzero entry for each row in an array """
            mask = arr!=0
            val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
            return np.where(mask.any(axis=axis), val, invalid_val)

        left = first_nonzero(mask, axis=1, invalid_val=-1)
        right = last_nonzero(mask, axis=1, invalid_val=-1)
        # print (left.shape)
        # find topmost row with at least 50 nonzero entries
        for idx, x in np.ndenumerate(left):   
            if x>-1:
                if right[idx] > x + 50:
                    goalx = math.floor(((x + right[idx]) / 2))
                    goaly = idx[0]
                    break
        intrvl = math.floor((400 - goaly)/3)
        waypnty = [400,400 - intrvl,400 - 2 * intrvl,goaly]
        waypntx1 = math.floor((left[waypnty[1]] + right[waypnty[1]]) / 2)
        waypntx2 = math.floor((left[waypnty[2]] + right[waypnty[2]]) / 2)
        waypntx = [105,waypntx1,waypntx2,goalx]
        return waypntx,waypnty

    def run(self,mask,img_count):
        if img_count > self.img_count:
            # waypoints
            self.waypntx,self.waypnty = self.setgoal(mask)
            # path
            self.rax, self.ray, self.ryaw = self.calc_spline_course(self.waypntx,self.waypnty)
            # speed profile
            self.speed_profile =  self.calc_speed_profile(self.ryaw)
        self.img_count = img_count
        return self.waypntx,self.waypnty,self.rax,self.ray,self.ryaw,self.speed_profile


class PlanMap(object):
    '''
    Create an image from the mask  
    with overlayed velocity and plan planning info
    '''
    def __init__(self, cfg):
        self.cfg = cfg
        self.fill = np.zeros((2,400,200),dtype=np.uint8)
        self.x = 105.
        self.y = 400.

    
    def run(self,mask,cx,cy,velturn,velfwd,rx,ry,delta,steeringangle,accel,steering,throttle):
        cax = math.floor(cx)
        cay = math.floor(cy)
        rax = np.empty_like(rx, dtype=np.int64)
        np.floor(rx, rax,casting="unsafe")
        ray = np.empty_like(ry, dtype=np.int64)
        np.floor(ry, ray,casting="unsafe")
        raxy = np.stack((rax,ray),axis=-1).reshape((-1,1,2))
        redmask = cv2.cvtColor(mask*200,cv2.COLOR_GRAY2RGB)
        #cv2.polylines(redmask,[waypntxy],False,(255,0,0),3)
        cv2.polylines(redmask,[raxy],False,(255,255,0),3)
        if steering is not None:
            steeringtxt = "{:f}".format(steering)
        else: 
            steeringtxt = ' '
        if throttle is not None:
            throttletxt = "{:f}".format(throttle)
        else: 
            throttletxt = ' '
        if steeringangle is not None:
            correctiontxt = "{:.1f}".format((steeringangle))
        else: 
            correctiontxt = ' '
        if accel is not None:
            acceltxt = "{:.1f}".format(accel)
        else: 
            acceltxt = ' '
        lines = correctiontxt + '\n' + acceltxt + '\n' + steeringtxt + '\n' + throttletxt
        dk.utils.draw_text(redmask,text=lines,uv_top_left=(120,200))
        target = delta - math.pi
        dy = math.floor(cay + (math.sin(target) * 100))
        dx = math.floor(cax + (math.cos(target) * 100))
        cv2.arrowedLine(redmask,(cax,cay),(dx,dy),(0, 255, 255), 2, cv2.LINE_AA, 0, 0.1)
        ex = math.floor(cax + (velturn*100.0))
        ey = math.floor(cay + (velfwd*100.0))
        cv2.arrowedLine(redmask,(cax,cay),(ex,ey),(0, 255, 0), 2, cv2.LINE_AA, 0, 0.1)
        # print(f'arrows ({cax},{cay}) ({ex},{ey}) and ({cax},{cay}) ({dx},{dy})')
        return redmask

 