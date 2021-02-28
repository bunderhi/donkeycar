'''
File: planner.py
Parts for path planning and tracking
'''
import numpy as np
import cv2
import scipy.interpolate as scipy_interpolate
from math import floor

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


    def run(self,img):

        original = self.reverse(img)
        undistorted_img = self.undistort(original)
        redm = cv2.cvtColor(undistorted_img,cv2.COLOR_RGB2GRAY).reshape(800,848)
        birdseye_mask = self.warpperspective(redm)
        return birdseye_mask

class PlanPath(object):
    """
    Determine a goal and a travel path 
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.n_path_points = 10

    
    def approximate_b_spline_path(self,x: list, y: list, degree: int = 3) -> tuple:
        """
        approximate points with a B-Spline path
        :param x: x position list of approximated points
        :param y: y position list of approximated points
        :param n_path_points: number of path points
        :param degree: (Optional) B Spline curve degree
        :return: x and y position list of the result path
        """
        t = range(len(x))
        x_tup = scipy_interpolate.splrep(t, x, k=degree)
        y_tup = scipy_interpolate.splrep(t, y, k=degree)

        x_list = list(x_tup)
        x_list[1] = x + [0.0, 0.0, 0.0, 0.0]

        y_list = list(y_tup)
        y_list[1] = y + [0.0, 0.0, 0.0, 0.0]

        ipl_t = np.linspace(0.0, len(x) - 1, self.n_path_points)
        rx = scipy_interpolate.splev(ipl_t, x_list)
        ry = scipy_interpolate.splev(ipl_t, y_list)

        return rx, ry

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

        goal = None
        left = first_nonzero(mask, axis=1, invalid_val=-1)
        right = last_nonzero(mask, axis=1, invalid_val=-1)
        # print (left.shape)
        # find topmost row with at least one nonzero entry
        for idx, x in np.ndenumerate(left):   
            if x>-1:
                #print(idx[0],x,right[idx])
                if right[idx] > x:
                    goalx = floor(((x + right[idx]) / 2))
                    goaly = idx[0]
                    # print ("Goal",goalx,goaly)
                    break
        
        waypnty = np.linspace(goaly,400,num=4,dtype=np.uint8)        
        waypntx1 = floor(((left[waypnty[1]] + right[waypnty[1]]) / 2))
        waypntx2 = floor(((left[waypnty[2]] + right[waypnty[2]]) / 2))
        waypntx = [goalx,waypntx1,waypntx2,105]
        return waypntx,waypnty

    def run(self,mask):
        waypntx,waypnty = self.setgoal(mask)
        rax, ray = self.approximate_b_spline_path(waypntx,waypnty)
        return waypntx,waypnty,rax,ray


class PlanMap(object):
    '''
    Create an image from the mask  
    with overlayed velocity and plan planning info
    '''
    def __init__(self, cfg):
        self.cfg = cfg
        self.fill = np.zeros((2,400,200),dtype=np.uint8)
    
    def draw_text(self,
        img,
        *,
        text,
        uv_top_left,
        color=(255, 255, 255),
        fontScale=0.5,
        thickness=1,
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        outline_color=(0, 0, 0),
        line_spacing=1.0,
    ):
        """
        Draws multiline with an outline.
        """
        assert isinstance(text, str)
        uv_top_left = np.array(uv_top_left, dtype=float)
        assert uv_top_left.shape == (2,)
        for line in text.splitlines():
            (w, h), _ = cv2.getTextSize(
                text=line,
                fontFace=fontFace,
                fontScale=fontScale,
                thickness=thickness,
            )
            uv_bottom_left_i = uv_top_left + [0, h]
            org = tuple(uv_bottom_left_i.astype(int))
            if outline_color is not None:
                cv2.putText(
                    img,
                    text=line,
                    org=org,
                    fontFace=fontFace,
                    fontScale=fontScale,
                    color=outline_color,
                    thickness=thickness * 3,
                    lineType=cv2.LINE_AA,
                )
            cv2.putText(
                img,
                text=line,
                org=org,
                fontFace=fontFace,
                fontScale=fontScale,
                color=color,
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )
            uv_top_left += [0, h * line_spacing]

    
    def run(self,mask,velfwd,velturn,waypntx,waypnty,rax,ray):
        waypntxy = np.stack((waypntx,waypnty),axis=-1).reshape((-1,1,2))
        raxy = np.stack((rax,ray),axis=-1).reshape((-1,1,2))
        redm = mask.reshape(1,400,200)
        redmask = np.vstack((self.fill,redm*255)).transpose(1,2,0)

        vx = "{:.1f}".format(velfwd *100.0)
        vy = "{:.1f}".format(velturn *100.0)
        lines = vx + '\n' + '\n' + vy
        self.draw_text(redmask,text=lines,uv_top_left=(120,240))
        cv2.polylines(redmask,[waypntxy],False,(0,255,255))
        cv2.polylines(redmask,[raxy],False,(255,255,0))
        return redmask

