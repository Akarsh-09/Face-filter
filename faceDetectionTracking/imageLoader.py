import mediapipe as mp
import numpy as np
import cv2 as cv

def openCamera():
    # Use OpenCV’s VideoCapture to start capturing from the webcam.
    cap = cv.VideoCapture(0)

    return cap

def imgLoader(cap):
    if cap is None or not cap.isOpened():
        print("Cannot open camera")
        return None, 0

    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        return None, 0

    return frame, ret