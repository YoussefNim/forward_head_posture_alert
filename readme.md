# Forward Head Posture Alert - v1

## Overview
This project aims to detect a user's posture using a webcam and alert them if they have bad posture (forward head posture). The application uses :
- **Mediapipe** for pose detection.
- **OpenCV** for face detection and visualizations.
- **winsound** and **threading** to play a sound (in a loop).
  
When the user's posture is detected as bad, the application plays an alert sound to prompt them to correct their posture. 

The program captures webcam feed and continuously monitors the user's pose. It uses the position of the ears and shoulders to determine if the posture is good or bad. If the user's posture is detected as bad, the program plays an alert sound. Once the posture is corrected, the sound stops.

## Current Features
- **Posture Detection**: The program analyzes the user's posture in real-time using **Mediapipe Pose**.
- **Face Detection**: **Haar Cascade classifiers** are used to detect faces (frontal and profile) to estimate ear positions when the pose landmarks are not detected.
- **Alert System**: An alert sound is played when the user has bad posture, and the sound stops when good posture is detected.
- **Real-time Feedback**: Posture status (good or bad) is displayed on the screen, along with colored visual indicators.

## Limitations
- **Sound Looping Issue**: The sound alert does not loop correctly in its current version. The sound plays once when bad posture is detected, but doesn't restart or loop until the posture changes.
- **Accuracy of Ear Estimation**: If the Mediapipe pose detection fails to detect ears, the program falls back to using the Haar Cascade face detection, which may not always give accurate ear positions, especially for side profiles.

## Requirements
- Python 3.x
- `mediapipe` library for pose detection
- `opencv-python` library for computer vision tasks
- `threading` and `winsound` libraries for sound control

### Install the required dependencies
To install the required dependencies, run:

```bash
pip install mediapipe opencv-python threading winsound