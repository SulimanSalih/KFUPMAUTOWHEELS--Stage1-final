# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : File Description and Imports

"""
vehicle_control.py

Skills acivity code for vehicle control lab guide.
Students will implement a vehicle speed and steering controller.
Please review Lab Guide - vehicle control PDF
"""
import os
import signal
import numpy as np
from threading import Thread
import time
import torch
import copy
import cv2
import pyqtgraph as pg
from pal.products.qcar import QCarRealSense
from hal.utilities.image_processing import ImageProcessing
from pal.products.qcar import QCar, QCarGPS, IS_PHYSICAL_QCAR
from pal.utilities.scope import MultiScope
from pal.utilities.math import wrap_to_pi
from hal.products.qcar import QCarEKF
from hal.products.mats import SDCSRoadMap
import pal.resources.images as images
from scipy import interpolate
import os
import signal
import numpy as np
from threading import Thread
from pal.utilities.scope import MultiScope
from pal.utilities.math import wrap_to_pi
from hal.products.qcar import QCarEKF
from hal.products.mats import SDCSRoadMap
import pal.resources.images as images
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
# model = YOLO('yolov8s.pt' )
# coneModel = YOLO('Cone.pt' )

# import rclpy
# from rclpy.node import Node
# from rclpy.qos import QoSProfile
# from rclpy.parameter import Parameter
# from rcl_interfaces.msg import SetParametersResult

# from geometry_msgs.msg import Vector3Stamped, PoseStamped
# from .submodules.geometry import euler_from_quaternion


#================ Experiment Configuration ================
# ===== Timing Parameters
# - tf: experiment duration in seconds.
# - startDelay: delay to give filters time to settle in seconds.
# - controllerUpdateRate: control update rate in Hz. Shouldn't exceed 500
tf = 300
startDelay = 1
start_part_1 = 3
controllerUpdateRate = 400

# ===== Speed Controller Parameters
# - v_ref: desired velocity in m/s
# - K_p: proportional gain for speed controller
# - K_i: integral gain for speed controller
v_ref = 0.7
K_p = 5
K_i = 0.2
K_d = 0
# Map_fully_updated = False # This is to check if the map is fully updated with all obstacles the adjustments to waypoints
# car_passed_track_once = False # This is to check if the car has passed the track once (don't update waypoints again)
# ===== Steering Controller Parameters
# - enableSteeringControl: whether or not to enable steering control
# - K_stanley: K gain for stanley controller
# - nodeSequence: list of nodes from roadmap. Used for trajectory generation.
enableSteeringControl = True
K_stanley = 1.2
nodeSequence = [10,2,4, 20,22,10]


#endregion
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : Initial setup
if enableSteeringControl:
    roadmap = SDCSRoadMap(leftHandTraffic=False)
    waypointSequence = roadmap.generate_path(nodeSequence)
    o_wp_sequence = waypointSequence
    initialPose = roadmap.get_node_pose(nodeSequence[0])
    print("type : ", type(waypointSequence))
else:
    initialPose = [0, 0, 0]

# if not IS_PHYSICAL_QCAR:
#     import qlabs_setup
#     qlabs_setup.setup(
#         initialPosition=[initialPose[0], initialPose[1], 0],
#         initialOrientation=[0, 0, initialPose[2]]
#     )

# Used to enable safe keyboard triggered shutdown
global KILL_THREAD
KILL_THREAD = False
global CONTRL_LOOP_AVOID_OBSTACLE
CONTRL_LOOP_AVOID_OBSTACLE = False
def sig_handler(*args):
    global KILL_THREAD
    KILL_THREAD = True
signal.signal(signal.SIGINT, sig_handler)
#endregion

class SpeedController:

    def __init__(self, kp=0, ki=0, kd=0):
        self.maxThrottle = 0.3

        self.kp = kp
        self.ki = ki
        self.kd = kd
        
        self.prev_e = 0
        
        

        self.ei = 0
        

    # ==============  SECTION A -  Speed Control  ====================
    def update(self, v, v_ref, dt):
        
        e = v_ref - v
        self.ei += dt*e
        ed = (e - self.prev_e) / dt if dt != 0 else 0
        self.prev_e = e
        

        
        return np.clip(
            self.kp*e + self.ki*self.ei+ self.kd * ed,
            -self.maxThrottle,
            self.maxThrottle
        )
        
        return 0

class SteeringController:

    def __init__(self, waypoints, k=1, cyclic=True):
        self.maxSteeringAngle = np.pi/6
        self.avoiding = False
        self.o_wp = waypoints.copy()
        self.wp = waypoints.copy()
        self.N = len(waypoints[0, :])
        self.wpi = 0
        self.sp = 0

        self.k = k
        self.cyclic = cyclic

        self.p_ref = (0, 0)
        self.th_ref = 0

    
    def update_waypoints(self, current_waypoint):
        # self.wp = waypoints
        # self.N = len(waypoints[0, :])
        f = np.mod(current_waypoint, self.N-1)
        mul = [0,0]
        diff_x = abs(self.wp[0,f +1 ]) - abs(self.wp[0,f])
        print(" Difference in x : ", diff_x)
        diff_y = abs(self.wp[1,f +1 ]) - abs(self.wp[1,f])
        print(" Difference in y : ", diff_y)
        
        
        if diff_x == 0:
            if self.wp[0,f ] > 0:
                print("Car is moving in positive y direction.")
                mul = [-1, 0]  # Shift on X axis +
            else:
                print("Car is moving in negative y direction.")
                mul = [-1, 0]  # Shift on Y axis -
        elif diff_y == 0 :
            if self.wp[1,f ] > 0:
                print("Car is moving in negative x.")
                mul = [0, -1]  # Shift on Y axis +
                # mul = [0, -1]  # Shift on X axis -
            else:
                print("Car is moving in positive x.")
                mul = [0, +1]  # Shift on X axis +
        else:
            print("car is moving in positive x and y direction.")
            mul = [0, 0]
        # #     mul = [1,0]
        # if diff_x > 0:
        #     if diff_y >= 0:
        #         print("Car is moving in positive x and positive y direction.")
        #         mul = [1, 1]  # Shift on Y axis +
        #     else:
        #         print("Car is moving in positive x and negative y direction.")
        #         mul = [-1, 0]  # Shift on Y axis -
        # else:
        #     if diff_y >= 0:
        #         print("Car is moving in negative x and positive y direction.")
        #         mul = [-1, 0]  # Shift on X axis -
        #     else:
        #         print("Car is moving in negative x and negative y direction.")
        #         mul = [1, 0]  # Shift on X axis +
        
        if self.avoiding == False:
            
            next_wp = np.mod(f + 90, self.N) 
            third_wp = np.mod(next_wp + 100,self.N)
            final_wp = np.mod(third_wp + 20,self.N)
            
            
            self.wp[0,f ] +=0.31* mul[0] 

            self.wp[1,f ] +=0.31* mul[1]
            

            self.wp[0,next_wp ] +=0.26 * mul[0]     
            self.wp[1,next_wp ] +=0.26 * mul[1] 

            p_car_start2 = [self.wp[0,f ], self.wp[1,f ]] # Car position or waypoint nearest to car in path
            p_cone2 = [self.wp[0,next_wp ], self.wp[1,next_wp ]] # detected cone position or nearest waypoint with Added Difference in y
            p_grace2 = [self.o_wp[0,third_wp ], self.wp[1,third_wp ]] # how long/flat the arc is 
            p_end2 =  [self.wp[0,final_wp ], self.wp[1,final_wp]] # last point connect to orignal path
            nodes2 = np.array( [ p_car_start2, p_cone2 ,p_grace2 ,p_end2] )

            x2 = nodes2[:,0]
            y2 = nodes2[:,1]
            
            tck2,u2     = interpolate.splprep( [x2,y2], k=3 ,s = 1 )
            unew2 = np.linspace(0, 1, 211)

            # Evaluate the spline at these points
            out2 = interpolate.splev(unew2, tck2)
            # print(out[1])
            self.wp[0,f:final_wp+1 ] = out2[0]
            self.wp[1,f:final_wp+1 ] = out2[1]
            print(" Old Waypoint index fixed : ", self.o_wp[0,f],"  "  , self.o_wp[1,f])

            print(" New Waypoint index fixed : ", self.wp[0,f],"  "  , self.wp[1,f])
            # self.wp[:,:] = self.o_wp[:,:]
            self.avoiding = True
            

    
    # ==============  SECTION B -  Steering Control  ====================
    
    
    def update(self, p, th, speed):
        wp_1 = 0.98*self.wp[:, np.mod(self.wpi, self.N-1)]
        wp_2 = 0.98*self.wp[:, np.mod(self.wpi+1, self.N-1)]
        
        # print("Waypoints 1 before : ", wp_1)
        # print("Waypoints 2 : before  ", wp_2)

        # wp_2[0] +=1
        # print(self.wpi)
        # print("Waypoints 1 : ", wp_1)
        # print("Waypoints 2 : ", wp_2)
        v = wp_2 - wp_1
        # print("Waypoints value difference : ", v)
       
        v_mag = np.linalg.norm(v)
        try:
            v_uv = v / v_mag
        except ZeroDivisionError:
            return 0

        tangent = np.arctan2(v_uv[1], v_uv[0])

        s = np.dot(p-wp_1, v_uv)

        if s >= v_mag:
            if  self.cyclic or self.wpi < self.N-2:
                self.wpi += 1

        ep = wp_1 + v_uv*s
        ct = ep - p
        dir = wrap_to_pi(np.arctan2(ct[1], ct[0]) - tangent)

        ect = 0.9*np.linalg.norm(ct) * np.sign(dir)
        psi = wrap_to_pi(tangent-th)

        self.p_ref = ep
        self.th_ref = tangent

        return np.clip(
            wrap_to_pi(psi + np.arctan2(self.k*ect, speed)),
            -self.maxSteeringAngle,
            self.maxSteeringAngle)
        
        return 0

def controlLoop():
    #region controlLoop setup
    global KILL_THREAD
    global CONTRL_LOOP_AVOID_OBSTACLE
    u = 0
    delta = 0
    currently_avoiding = False
    # used to limit data sampling to 10hz
    countMax = controllerUpdateRate / 10
    count = 0
    #endregion
    
    #region Controller initialization
    speedController = SpeedController(
        kp=K_p,
        ki=K_i,
        kd=K_d

    )
    if enableSteeringControl:
        steeringController = SteeringController(
            waypoints=waypointSequence,
            k=K_stanley
        )
    #endregion

    #region QCar interface setup
    qcar = QCar(readMode=1, frequency=controllerUpdateRate)
    if enableSteeringControl:
        ekf = QCarEKF(x_0=initialPose)
        gps = QCarGPS(initialPose=initialPose)
    else:
        gps = memoryview(b'')
    #endregion

    with qcar, gps:
        t0 = time.time()
        t=0


        while (t < tf+startDelay) and (not KILL_THREAD):
            #region : Loop timing update
            tp = t
            t = time.time() - t0
            dt = t-tp
            #endregion

            #region : Read from sensors and update state estimates
            qcar.read()
            if enableSteeringControl:
                if gps.readGPS():
                    y_gps = np.array([
                        gps.position[0],
                        gps.position[1],
                        gps.orientation[2]
                    ])
                    ekf.update(
                        [qcar.motorTach, delta],
                        dt,
                        y_gps,
                        qcar.gyroscope[2],
                    )
                else:
                    ekf.update(
                        [qcar.motorTach, delta],
                        dt,
                        None,
                        qcar.gyroscope[2],
                    )

                x = ekf.x_hat[0,0]
                y = ekf.x_hat[1,0]
                th = ekf.x_hat[2,0]
                p = ( np.array([x, y])
                    + np.array([np.cos(th), np.sin(th)]) * 0.2)
            v = qcar.motorTach
            #endregion

            #region : Update controllers and write to car
            if t < start_part_1:
                v_ref = 0.5
            else:
                v_ref = 0.7
            if t < startDelay:
                u = 0
                delta = 0

            else:
                #region : Speed controller update
                u = speedController.update(v, v_ref, dt)
                #endregion

                #region : Steering controller update
                if enableSteeringControl:
                    delta = steeringController.update(p, th, v)
                    
                    if CONTRL_LOOP_AVOID_OBSTACLE == True:
                        steeringController.sp = copy.copy(steeringController.wpi) 
                        CONTRL_LOOP_AVOID_OBSTACLE = False
                    if steeringController.avoiding == False and steeringController.wpi == steeringController.sp:
                        # CONTRL_LOOP_AVOID_OBSTACLE = False
                        print(" avoiding something updating waypoints at : ", np.mod(steeringController.wpi, steeringController.N))
                        steeringController.update_waypoints(steeringController.sp)
                        referencePath.setData(steeringController.wp[0, :],steeringController.wp[1, :])
                    #     steeringController.avoiding == True
                    # print(" Waypoint index fixed : ", steeringController.sp)
                    # print(" Waypoint index current : ", steeringController.wpi)
                    #print("avoiding  : ", steeringController.avoiding)
                    if steeringController.avoiding == True and steeringController.wpi >= steeringController.sp+221:
                        print(" Avoiding done updating waypoints at : ", np.mod(steeringController.wpi, steeringController.N))
                        steeringController.avoiding = False
                        steeringController.wp = steeringController.o_wp.copy()
                        # print(" Modified waypoints : ", steeringController.wp[0:sp])
                        # print(" Original waypoints : ", steeringController.o_wp[0:sp])
                        # steeringController.wp = steeringController.o_wp 
                        # print(" Not avoiding anything resetting map waypoints ")
                        # steeringController.wp = steeringController.o_wp
             
                else:
                    delta = 0
                #endregion

            qcar.write(u, delta)
            #endregion
         
            #region : Update Scopes
            count += 1
            if count >= countMax and t > startDelay:
                t_plot = t - startDelay

                # # Speed control scope
                # speedScope.axes[0].sample(t_plot, [v, v_ref])
                # speedScope.axes[1].sample(t_plot, [v_ref-v])
                # speedScope.axes[2].sample(t_plot, [u])

                # Steering control scope
                # if enableSteeringControl:
                    # steeringScope.axes[4].sample(t_plot, [[p[0],p[1]]])

                    # p[0] = ekf.x_hat[0,0]
                    # p[1] = ekf.x_hat[1,0]

                    # x_ref = steeringController.p_ref[0]
                    # y_ref = steeringController.p_ref[1]
                    # th_ref = steeringController.th_ref

                    # x_ref = gps.position[0]
                    # y_ref = gps.position[1]
                    # th_ref = gps.orientation[2]

                    # steeringScope.axes[0].sample(t_plot, [p[0], x_ref])
                    # steeringScope.axes[1].sample(t_plot, [p[1], y_ref])
                    # steeringScope.axes[2].sample(t_plot, [th, th_ref])
                    # steeringScope.axes[3].sample(t_plot, [delta])


                    # arrow.setPos(p[0], p[1])
                    # arrow.setStyle(angle=180-th*180/np.pi)

                count = 0
            #endregion
            continue

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : Setup and run experiment
if __name__ == '__main__':

    #region : Setup scopes
    if IS_PHYSICAL_QCAR:
        fps = 10
    else:
        fps = 30
    # Scope for monitoring speed controller
    # speedScope = MultiScope(
    #     rows=3,
    #     cols=1,
    #     title='Vehicle Speed Control',
    #     fps=fps
    # )
    # speedScope.addAxis(
    #     row=0,
    #     col=0,
    #     timeWindow=tf,
    #     yLabel='Vehicle Speed [m/s]',
    #     yLim=(0, 1)
    # )
    # speedScope.axes[0].attachSignal(name='v_meas', width=2)
    # speedScope.axes[0].attachSignal(name='v_ref')

    # speedScope.addAxis(
    #     row=1,
    #     col=0,
    #     timeWindow=tf,
    #     yLabel='Speed Error [m/s]',
    #     yLim=(-0.5, 0.5)
    # )
    # speedScope.axes[1].attachSignal()

    # speedScope.addAxis(
    #     row=2,
    #     col=0,
    #     timeWindow=tf,
    #     xLabel='Time [s]',
    #     yLabel='Throttle Command [%]',
    #     yLim=(-0.3, 0.3)
    # )
    # speedScope.axes[2].attachSignal()

    # Scope for monitoring steering controller
    if enableSteeringControl:
        steeringScope = MultiScope(
            rows=4,
            cols=2,
            title='Vehicle Steering Control',
            fps=fps
        )

        steeringScope.addAxis(
            row=0,
            col=0,
            timeWindow=tf,
            yLabel='x Position [m]',
            yLim=(-2.5, 2.5)
        )
        steeringScope.axes[0].attachSignal(name='x_meas')
        steeringScope.axes[0].attachSignal(name='x_ref')

        steeringScope.addAxis(
            row=1,
            col=0,
            timeWindow=tf,
            yLabel='y Position [m]',
            yLim=(-1, 5)
        )
        steeringScope.axes[1].attachSignal(name='y_meas')
        steeringScope.axes[1].attachSignal(name='y_ref')

        steeringScope.addAxis(
            row=2,
            col=0,
            timeWindow=tf,
            yLabel='Heading Angle [rad]',
            yLim=(-3.5, 3.5)
        )
        steeringScope.axes[2].attachSignal(name='th_meas')
        steeringScope.axes[2].attachSignal(name='th_ref')

        steeringScope.addAxis(
            row=3,
            col=0,
            timeWindow=tf,
            yLabel='Steering Angle [rad]',
            yLim=(-0.6, 0.6)
        )
        steeringScope.axes[3].attachSignal()
        steeringScope.axes[3].xLabel = 'Time [s]'

        steeringScope.addXYAxis(
            row=0,
            col=1,
            rowSpan=4,
            xLabel='x Position [m]',
            yLabel='y Position [m]',
            xLim=(-2.5, 2.5),
            yLim=(-1, 5)
        )

        im = cv2.imread(
            images.SDCS_CITYSCAPE,
            cv2.IMREAD_GRAYSCALE
        )

        steeringScope.axes[4].attachImage(
            scale=(-0.002035, 0.002035),
            offset=(1125,2365),
            rotation=180,
            levels=(0, 255)
        )
        steeringScope.axes[4].images[0].setImage(image=im)

        referencePath = pg.PlotDataItem(
            pen={'color': (85,168,104), 'width': 2},
            name='Reference'
        )
        steeringScope.axes[4].plot.addItem(referencePath)
       

        steeringScope.axes[4].attachSignal(name='Estimated', width=2)


        ###  ==============================
        # referencePath2 = pg.PlotDataItem(
        #     pen={'color': (185,92,154), 'width': 2},
        #     name='Reference2'
        # )
        # steeringScope.axes[4].plot.addItem(referencePath2)
       



        # waypointSequence[1,180 ] +=0.16
        # # waypointSequence[0,350 ] +=0.0001     


        # p_car_start = [waypointSequence[0,160 ], waypointSequence[1,160 ]] # Car position or waypoint nearest to car in path
        # p_cone = [waypointSequence[0,180 ], waypointSequence[1,180 ]] # detected cone position or nearest waypoint with Added Difference in y
        # p_grace = [waypointSequence[0,340 ], waypointSequence[1,340 ]] # how long/flat the arc is 
        # p_end =  [waypointSequence[0,370 ], waypointSequence[1,370 ]] # last point connect to orignal path
        # nodes = np.array( [ p_car_start, p_cone ,p_grace ,p_end] )

        # x1 = nodes[:,0]
        # y1 = nodes[:,1]
        # # print([x1,y1])
        # tck,u     = interpolate.splprep( [x1,y1], k=3 ,s = 0 )
        # unew = np.linspace(0, 1, 211)

        # # Evaluate the spline at these points
        # out = interpolate.splev(unew, tck)
        # # print(out[1])
        # waypointSequence[0,160:371 ] = out[0]
        # waypointSequence[1,160:371 ] = out[1]
       # ==========


        
        # waypointSequence[1,430 ] +=0.16
        # waypointSequence[0,550 ] +=0.0001     


        # p_car_start = [waypointSequence[0,400 ], waypointSequence[1,400 ]] # Car position or waypoint nearest to car in path
        # p_cone = [waypointSequence[0,430 ], waypointSequence[1,430 ]] # detected cone position or nearest waypoint with Added Difference in y
        # p_grace = [waypointSequence[0,490 ], waypointSequence[1,490 ]] # how long/flat the arc is 
        # p_end =  [waypointSequence[0,550 ], waypointSequence[1,550 ]] # last point connect to orignal path
        # nodes = np.array( [ p_car_start, p_cone ,p_grace ,p_end] )

        # x1 = nodes[:,0]
        # y1 = nodes[:,1]
        # # print([x1,y1])
        # tck2,u     = interpolate.splprep( [x1,y1], k=3 ,s = 0 )
        # unew2 = np.linspace(0, 1, 151)

        # # Evaluate the spline at these points
        # out2 = interpolate.splev(unew2, tck2)
        # print(out2.shape)
        # waypointSequence[0,400:551 ] = out2[0]
        # waypointSequence[1,400:551 ] = out2[1]
       # ==========



        referencePath.setData(waypointSequence[0, :],waypointSequence[1, :])

        ### ================================
        arrow = pg.ArrowItem(
            angle=180,
            tipAngle=60,
            headLen=10,
            tailLen=10,
            tailWidth=5,
            pen={'color': 'w', 'fillColor': [196,78,82], 'width': 1},
            brush=[196,78,82]
        )
        arrow.setPos(initialPose[0], initialPose[1])
        steeringScope.axes[4].plot.addItem(arrow)
    #endregion

    #region : Setup control thread, then run experiment
    controlThread = Thread(target=controlLoop)
    controlThread.start()
    COUNTER=0
    imageWidth  = 640
    imageHeight = 480
    myCam  = QCarRealSense(mode='RGB&DEPTH',
            frameWidthRGB=imageWidth,
            frameHeightRGB=imageHeight)
    t0 = time.time() 
    tstop=-1.0
    FLAG='pass'
    # coun
    roi_width = 190
    roi_height = 250
    roi_x = int((imageWidth - roi_width) / 2)
    roi_y = imageHeight - roi_height
    myCam  = QCarRealSense(mode='RGB&DEPTH',
                frameWidthRGB=imageWidth,
                frameHeightRGB=imageHeight)
    # model = YOLO('yolov8s.pt' )
    model = YOLO('Cone.pt')
    frame_count = 0
    consecutive_frames_threshold = 2
    # Dictionary to keep track of boxes and their consecutive frame counts
    boxes_dict = {}
    def update_boxes_dict(boxes_dict,class_id, box_coords):
        if class_id in boxes_dict:
            box, count = boxes_dict[class_id]
            boxes_dict[class_id] = (box_coords, count + 1)
        else:
            boxes_dict[class_id] = (box_coords, 1)

        

        return boxes_dict
    try:
        while True:
            myCam.read_RGB()
            inputRGB = myCam.imageBufferRGB
            # binaryImage3 = ImageProcessing.binary_thresholding(frame= inputRGB,
            #                         lowerBounds=np.array([0, 0, 0]),
            #                         upperBounds=np.array([200, 200, 200]))
            # cv2.imshow('binary mask Image 1', binaryImage3)
            # apply morphology close with horizontal rectangle kernel to fill horizontal gap
            # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (121,121))
            # morph2 = cv2.morphologyEx(binaryImage3, cv2.MORPH_CLOSE, kernel)
            



            # masked = cv2.bitwise_and(hsvBuf, hsvBuf, mask=binaryImage3)
            # masked_rgb = cv2.bitwise_and(inputRGB, inputRGB, mask=morph2)
            # cv2.imshow('Masked Image 1', masked_rgb)
            frame_count += 1
            if CONTRL_LOOP_AVOID_OBSTACLE == False:
                results = model(inputRGB,conf=0.3,verbose=False)  # return a list of Results objects
                for r in results:
                    # annotator = Annotator(myCam.imageBufferRGB)
                    boxes = r.boxes
                    for box in boxes:
                        xA, yA, xB, yB = box.xyxy[0]
                        # Check if the bounding box is completely inside the ROI
                        if roi_x < xA < xB < roi_x + roi_width and roi_y < yA < yB < roi_y + roi_height:
                            cv2.rectangle(inputRGB, (int(xA), int(yA)), (int(xB), int(yB)), (0, 255, 0), 2)
                            class_id = int(box.cls[0])  # Get the class of the box
                            print("Cone is within range")
                            # Check if box is fully inside ROI

                            boxes_dict = update_boxes_dict(boxes_dict, class_id, (xA, yA, xB, yB))
                            # check box area is greater than 1000
                            # if (xB - xA) * (yB - yA) > 0.2* (roi_width * roi_height) :
                            #     print("Cone is close enough")
                            if boxes_dict[class_id][1] >= 2:
                                v_ref = 0.4
                            # if boxes dict count is greater than 2 print the boxes
                            if boxes_dict[class_id][1] >= 2:
                                #print("Cone in ROI for enough frames to be an obstacle:", boxes_dict) 
                                
                                box_center_x = (xA + xB) / 2
                                box_center_y = (yA + yB) / 2
                                
                                # Calculate ROI center
                                roi_center_x = roi_x + roi_width / 2
                                roi_center_y = roi_y + roi_height / 2
                                
                                # Calculate distance between box center and ROI center
                                distance = ((box_center_x - roi_center_x)**2 + (box_center_y - roi_center_y)**2)**0.5
                                # Convert distance to a numerical value
                                if isinstance(distance, torch.Tensor):  # for PyTorch
                                    distance_value = distance.item()
                                elif isinstance(distance, tf.Tensor):    # for TensorFlow
                                    distance_value = distance.numpy()
                                else:
                                    distance_value = distance
                                print("Distance between box center and ROI center:", distance_value)   
                                if distance_value <= 65:
                                    # print("Obstacle detected. Avoiding obstacle.")
                                    CONTRL_LOOP_AVOID_OBSTACLE = True
                                    break
                        break
                        # c = box.cls
                        # annotator.box_label(b, model.names[int(c)])
                # cv2.rectangle(inputRGB, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 0, 255), 2)
                # cv2.imshow('YOLO V8 Detection', inputRGB)
                
                if frame_count >= 10:

                    frame_count = 0
                    boxes_dict = {}
                    # print("Resetting boxes_dict to empty.", boxes_dict)
                # cv2.imshow('YOLO V8 Detection', r.plot())
                cv2.waitKey(1)  
            # coneresults = Cone_model(myCam.imageBufferRGB,conf=0.6,verbose=False)  # return a list of Results objects
            # for c in coneresults :
            #     # cannotator = Annotator(myCam.imageBufferRGB)
            #     boxes = c.boxes
            #     cv2.imshow('YOLO V8 CONE/OBJECTS Detection', c.plot())
            #     cv2.waitKey(1)  
            # Remove boxes that have not been seen for at least consecutive_frames_threshold frames
            # boxes_dict = {k: v for k, v in boxes_dict.items() if v[1] >= consecutive_frames_threshold}
            # if boxes_dict != {}:
            #     print("Boxes in ROI:", boxes_dict)
            
    # except:
    #     print('OUTPUT')
    except Exception as e:
        print("An exception occurred:", e)
