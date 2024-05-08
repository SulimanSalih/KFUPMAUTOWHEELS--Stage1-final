from ultralytics import YOLO
import cv2
# Load a model
model = YOLO('yolov8s.pt' )  # pretrained YOLOv8n model with classes 9 and 11


import cv2
import numpy as np

# Convert HEX to RGB
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    lv = len(hex_color)
    return tuple(int(hex_color[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

# Convert RGB to HSV
def rgb_to_hsv(r, g, b):
    color = np.uint8([[[b, g, r]]])
    hsv_color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    return hsv_color[0][0]

# Load image
def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image {image_path}")
    return image

# Calculate brightness
def calculate_brightness(mask):
    return np.sum(mask) / np.count_nonzero(mask) if np.count_nonzero(mask) else 0

# Define the color ranges in HSV space
green_on_hsv = rgb_to_hsv(*hex_to_rgb("#78F569"))
green_off_hsv = rgb_to_hsv(*hex_to_rgb("#20712F"))
red_on_hsv = rgb_to_hsv(*hex_to_rgb("#FB6B51"))
red_off_hsv = rgb_to_hsv(*hex_to_rgb("#79414E"))

# Function to create a mask for a given color
def create_mask(hsv_image, color_hsv):
    lower_bound = np.array([color_hsv[0] - 10, max(color_hsv[1] - 40, 100), max(color_hsv[2] - 40, 100)])
    upper_bound = np.array([color_hsv[0] + 10, min(color_hsv[1] + 40, 255), min(color_hsv[2] + 40, 255)])
    return cv2.inRange(hsv_image, lower_bound, upper_bound)

# Process each image
def process_images(yoloimage):
    if yoloimage is None:
        return
    hsv_image = cv2.cvtColor(yoloimage, cv2.COLOR_BGR2HSV)
    results = {}
    masks = {
        'green_on': create_mask(hsv_image, green_on_hsv),
        'green_off': create_mask(hsv_image, green_off_hsv),
        'red_on': create_mask(hsv_image, red_on_hsv),
        'red_off': create_mask(hsv_image, red_off_hsv)
    }
    brightness = {color: calculate_brightness(mask) for color, mask in masks.items()}
    light_status = 'green' if brightness['green_on'] > brightness['green_off'] else 'red'
    # print(light_status)
    # results[image_path] = {
    #     'brightness': brightness,
    #     'status': light_status
    # }
    results= light_status        
    if results =='red':
        print("------------------------------"+"STOP CODE"+"-------------------------------")
    elif results =='green':
        print("------------------------------PASS------------------------------")
    else:
        print("----------------------------Nothing-----------------------------")

    # return results

# Image paths
# image_paths = ['green1.jpg','green2.jpg','red3.jpg']
# Test = load_image(image_paths[2])
# Process the images and get the results
# results = process_images(cropped_image)


# Run batched inference on a list of images
def mov_logic(image):
#---------------------------------------------------------------------------------------LOGIC
    results = model(image,classes=[9,11],conf=0.70)  # return a list of Results objects
    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        if boxes.cls[0].item()==9.0:
            print(boxes.cls[0].item())
            x1, y1, x2, y2 = map(int, boxes.xyxy[0])  # Get bounding box coordinates
            cropped_image = image[y1:y2, x1:x2]  #
            process_images(cropped_image)
        elif boxes.cls[0].item()==11.0:
            print("------------------------------"+"STOP CODE"+"-------------------------------")
        # masks = result.masks  # Masks object for segmentation masks outputs
        # keypoints = result.keypoints  # Keypoints object for pose outputs
        # probs = result.probs  # Probs object for classification outputs
        # result.show()  # display to screen
    # result.save(filename='result.jpg')  # save to disk
# print(model.names[int(result.Boxes)])
# cv2.imshow("tt", cropped_image)
# cv2.imwrite("test10.jpg", cropped_image)  # Save the cropped image to disk
image = cv2.imread('imt2.jpg')
mov_logic(image)