import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8l.pt' )  # pretrained YOLOv8n model with classes 9 and 11
# model = YOLO('yolov8s.pt' )  # pretrained YOLOv8n model with classes 9 and 11
# model = YOLO('best.pt')
# rjr ghtoe ghirfopremfjngiuyuo 
# import numpy as np

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

class TrafficLightLogic:
    def __init__(self):
        self.model = YOLO('yolov8s.pt')
        self.green_on_hsv = self.rgb_to_hsv(*self.hex_to_rgb("#78F569"))
        self.green2_on_hsv = self.rgb_to_hsv(*self.hex_to_rgb("#00ff00"))
        self.green_off_hsv = self.rgb_to_hsv(*self.hex_to_rgb("#20712F"))
        self.red_on_hsv = self.rgb_to_hsv(*self.hex_to_rgb("#FB6B51"))
        self.red_off_hsv = self.rgb_to_hsv(*self.hex_to_rgb("#79414E"))

    @staticmethod
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        lv = len(hex_color)
        return tuple(int(hex_color[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

    @staticmethod
    def rgb_to_hsv(r, g, b):
        color = np.uint8([[[b, g, r]]])
        hsv_color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        return hsv_color[0][0]

    @staticmethod
    def load_image(image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image {image_path}")
        return image

    @staticmethod
    def calculate_brightness(mask):
        return np.sum(mask) / np.count_nonzero(mask) if np.count_nonzero(mask) else 0

    def create_mask(self, hsv_image, color_hsv):
        lower_bound = np.array([color_hsv[0] - 10, max(color_hsv[1] - 40, 100), max(color_hsv[2] - 40, 100)])
        upper_bound = np.array([color_hsv[0] + 10, min(color_hsv[1] + 40, 255), min(color_hsv[2] + 40, 255)])
        return cv2.inRange(hsv_image, lower_bound, upper_bound)

    @staticmethod
    def disI(x1, y1, x2, y2, image):
        box_width = x2 - x1
        box_height = y2 - y1
        frame_area = image.shape[0] * image.shape[1]
        box_area = box_width * box_height
        distance = (box_area / frame_area) * 100
        return round(distance, 3)

    def process_images(self, yoloimage):
        if yoloimage is None:
            return None
        hsv_image = cv2.cvtColor(yoloimage, cv2.COLOR_BGR2HSV)
        masks = {
            'green_on': self.create_mask(hsv_image, self.green_on_hsv),
            'green2_on': self.create_mask(hsv_image, self.green2_on_hsv),
            'green_off': self.create_mask(hsv_image, self.green_off_hsv),
            'red_on': self.create_mask(hsv_image, self.red_on_hsv),
            'red_off': self.create_mask(hsv_image, self.red_off_hsv)
        }
        brightness = {color: self.calculate_brightness(mask) for color, mask in masks.items()}
        light_status = 'green' if brightness['green_on'] > brightness['green_off'] else 'red'
        light_status2 = 'green' if brightness['green2_on'] > brightness['green_off'] else 'red'
        if light_status == 'green' or light_status2 == 'green':
            return 'green'
        else:
            return 'red'

    def mainlogic(self, gflag, dis,image):
        if gflag == "green":
            return 'green'
        elif gflag == "red" and self.detect_horizontal_lines(image):
            print("red")
            print("----------------------------------")
            print(dis)
            return "red"
        elif gflag == "red" and  dis <= 0.29 and dis >= 0.200:
            print("red"+'------------------------'+str(dis))
            return "red"
        elif gflag == "stop" and dis >= 0.50:
            return "stop"
        else:
            return 'pass'

    def mov_logic(self, image):
        results = self.model(image, classes=[9, 11], conf=0.3, verbose=False)
        for result in results:
            boxes = result.boxes
            if not torch.equal(torch.tensor([]), boxes.cls):
                if boxes.cls[0].item() == 9.0:  # traffic light
                    x1, y1, x2, y2 = map(int, boxes.xyxy[0])  # Get bounding box coordinates
                    cropped_image = image[y1:y2, x1:x2]
                    dis = self.disI(x1, y1, x2, y2, image)
                    getflag = self.process_images(cropped_image)
                    return self.mainlogic(getflag, dis,image)
                elif boxes.cls[0].item() == 11.0:  # stop sign
                    x1, y1, x2, y2 = map(int, boxes.xyxy[0])  # Get bounding box coordinates
                    dis = self.disI(x1, y1, x2, y2, image)
                    getflag = "stop"
                    return self.mainlogic(getflag, dis,image)

    def detect_horizontal_lines(self,image):
        global y1
        y1=0
    #    image=cv2.imread(image)
    #    image =np.copy(image)
        image = image[300:400, 200:600]

        image= cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
        if (lines is None):
            return False
        else:
            detected_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1 + 1e-6)  # Calculate slope (avoiding division by zero)
                if abs(slope) < 0.1 and x2-x1 < 490 :  # Adjust the slope threshold as needed
                    # print ( x2-x1)
                    detected_lines.append((x1, y1, x2, y2))
                    # print("stop")

            for line in detected_lines:
                x1, y1, x2, y2 = line
                print(x2 , x1 ,y2 ,y1)
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
               
                if y1 >= 30:
                    print("OK DISTANCE FOEM THE STOPUNG LINE")
                    return (True)
                else:
                    return (False)
        cv2.imshow("Detected Lines", image)
        # return (print(y1))
            # pass

# Example Usage
# logic = TrafficLightLogic()
# image = logic.load_image('testpx.png')
# print(logic.mov_logic(image))


# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
# image = cv2.imread('testpx.png') #   gren traffic
# image = cv2.imread('imt1.jpg')#gren traffic
# image = cv2.imread('stop.jpg')# stop
# image = cv2.imread('im2.png') #red traffic
# image = cv2.imread('2im.jpg') #2 obj
# print(process_images(image))
# print(mov_logic(image))
# result= model(image,classes=[9,11],conf=0.5,verbose=False)  # return a list of Results objects
# for r in result:
#     # print(r.names)
#     cv2.imshow('image', r.show())