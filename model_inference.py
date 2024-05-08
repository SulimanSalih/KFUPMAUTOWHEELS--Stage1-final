from pal.products.qcar import QCarRealSense
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2
imageWidth  = 640
imageHeight = 480
myCam  = QCarRealSense(mode='RGB&DEPTH',
            frameWidthRGB=imageWidth,
            frameHeightRGB=imageHeight)
model = YOLO('yolov8s.pt' )
Cone_model = YOLO('Cone.pt')
try:
    while True:
        myCam.read_RGB()
        results = model(myCam.imageBufferRGB,classes=[0,9,11,17,57,72],conf=0.6,verbose=False)  # return a list of Results objects
        for r in results:
            # annotator = Annotator(myCam.imageBufferRGB)
            boxes = r.boxes
            # for box in boxes:
            #     b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
            #     c = box.cls
            #     annotator.box_label(b, model.names[int(c)])
            cv2.imshow('YOLO V8 Detection', r.plot())
            cv2.waitKey(1)  
        coneresults = Cone_model(myCam.imageBufferRGB,conf=0.6,verbose=False)  # return a list of Results objects
        for c in coneresults :
            # cannotator = Annotator(myCam.imageBufferRGB)
            boxes = c.boxes
            cv2.imshow('YOLO V8 CONE/OBJECTS Detection', c.plot())
            cv2.waitKey(1)  
except:
    print('OUTPUT')

