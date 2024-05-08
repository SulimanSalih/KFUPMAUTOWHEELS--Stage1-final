from ultralytics import YOLO
import cv2
import torch
model = YOLO('Cone.pt')

# Load a model
def dis(xyxy,image):
    x1, y1, x2, y2 = xyxy
    # print(x1)
    box_width = x2 - x1
    box_height = y2 - y1
    frame_width = image.shape[1]
    frame_height = image.shape[0]
    frame_area = frame_width*frame_height
    box_area= box_width * box_height
    distance = (box_area / frame_area) * 100
    return round(distance,1)

def conedetact(image):
    disv=0
    results = model(image,conf=0.65,verbose=False)  # return a list of Results objects
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        if not torch.equal(torch.tensor([]),boxes.cls):    
                if boxes.cls[0].item()==0.0: # traffic ligth
                    print('333333333333333333333333333333333')
                    x1, y1, x2, y2 = map(int, boxes.xyxy[0])  # Get bounding box coordinates
                    cropped_image = image[y1:y2, x1:x2]  #
                    # print(dis([x1,y1,x2,y2],image))
                    disv=dis([x1,y1,x2,y2],image)
                    # disv == 
   # if disv >=1.2 and disv <=1.9:
    if disv == 0.9 or disv == 1.3 or disv == 1.2 or disv ==1.1 or disv ==1.0:
        print('7777777777777777777777777')
        return 'cone'
    else:
        return 'pass'



















    # masks = result.masks  # Masks object for segmentation masks outputs
    # keypoints = result.keypoints  # Keypoints object for pose outputs
    # probs = result.probs  # Probs object for classification outputs
    # obb = result.obb  # Oriented boxes object for OBB outputs
    # result.show()  # display to screen
    # # result.save(filename='result.jpg')  # save to disk







