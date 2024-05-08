# KFUPM AutoWheels
# [Competition Submission Stage 1](https://youtu.be/iLufV5nQwpI)
![image](https://github.com/SulimanSalih/KFUPMAUTOWHEELS--Stage1/assets/108358496/d6eec892-a7b1-4839-b83c-e7ed3c3dcf50)

# Competition Submission Read Me

## Overview
Hey there! We're thrilled to share our submission for the Self-Driving Challenge. As students from KFUPM, we're pumped to dive into the world of autonomous vehicles and show what we've got!
Our project is all about having fun while exploring Control Systems stuff, especially the cool bits about making split-second decisions for our self-driving car. We're all about that perfect balance of speed and staying on track.
You will find in this repo, our competition submission! This package contains the code and a recorded video of our execution for the competition. Below are some key points to note regarding our submission:

## Testing Environment
- The testing was conducted on version 2.16 of Quanser Interactive Labs Software.
- We utilized the primary Quanser computer in the Qcar Lab for testing.
- YOLOv8s was used for detecting the obstacle (cones) and detecting the stop sign.


## Results
- The results of our testing met all competition requirements.
- Smooth Qcar movement was observed during the execution.
- We tuned the Controllers params for better navigation, balancing speed with safety.
- For Traffic Regulations: Stop signs and traffic lights are handled according to competition requirments, thanks to our tuned algorithms.
- Obstacle Avoidance: We tested Our system and reached to a level where it navigates obstacles with agile maneuvers.

## Variations
- It's important to acknowledge that variations may occur when utilizing different PCs or versions of the QLab Software.

## Development Process
- Our team took a systematic approach to meet each competition requirement individually.
- We then integrated the entire codebase for a cohesive solution.

## Next Steps
- We achieved satisfactory results and eagerly anticipate applying our solution to the physical Qcar in Stage two of the competition.

## Additional Installation Instructions
To run the code successfully, please ensure you have the following dependencies installed:
- [YOLOv8](https://docs.ultralytics.com/quickstart/)
- torch
- numpy

**- cv2
You can install YOLOv8 and its dependencies from the following link:
[YOLOv8 and models weights](https://drive.google.com/drive/folders/1a5cZfxemTJMMAq51wtCSkZUDMenIBbD6?usp=sharing)**

## IMPORTANT 
- We Used 2 seperate Yolov8 models, the two files are part of this repo but you need to download them from GDrive link above, it is important that these two .pt files are there in same root of the folder of the SDCS_Main.py

## File Descriptions:
1. SDCS_Main
   This file is the main script responsible for motion and speed control through PID control. It contains the AI details and practicalities based on distance, colors, and subsequently stopping or moving away.
2. QCar_obstacle_detection :
   This file is our work on making the Qcar see and avoid obstacles on the road, we trained a simple YoloV8s model so that it detects the small cones and adjust waypoints to avoid hitting the cones.

3. model_inference
   This file handles the inference for all road signs (such as signals, stop signs, pedestrian crossings, etc.) as well as a file for recognizing people and objects.

Extra : the following files: movement.py, cone.py, YOLOLOGIC.py
   These are a set of files used or to be called upon to perform various independent tasks they are not neccesary for the main code to run, it was only for testing our new methods before using them in the main file.


  
Messing around with Control Systems has been a blast. It's like being in a video game, but way cooler 'cause it's real life!
This challenge has taught us a ton about making split-second decisions, which we're sure will come in handy down the road.
Plus, teaming up with folks from all majors has been a blast. Who knew self-driving cars could bring so many cool people together?
