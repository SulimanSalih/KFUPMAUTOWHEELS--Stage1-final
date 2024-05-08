# KFUPM AutoWheels
# [Competition Submission Stage 1](https://youtu.be/iLufV5nQwpI)
![image](https://github.com/SulimanSalih/KFUPMAUTOWHEELS--Stage1/assets/108358496/d6eec892-a7b1-4839-b83c-e7ed3c3dcf50)

# Competition Submission Read Me

## Overview
Welcome to our competition submission! This package contains the code and a recorded video of our execution for the competition. Below are some key points to note regarding our submission:

## Testing Environment
- The testing was conducted on version 2.16 of Quanser Interactive Labs Software.
- We utilized the primary cQuanser computer in the Qcar Lab for testing.

## Results
- The results of our testing met all competition requirements.
- Smooth Qcar movement was observed during the execution.

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
## File Descriptions:
1. SDCS_Main
   This file is the main script responsible for motion and speed control through PID control. It contains the AI details and practicalities based on distance, colors, and subsequently stopping or moving away.

2. model_inference
   This file handles the inference for all road signs (such as signals, stop signs, pedestrian crossings, etc.) as well as a file for recognizing people and objects.

3. movement.py, cone.py, YOLOLOGIC.py
   These are a set of files used or to be called upon to perform various tasks.
