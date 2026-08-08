import mediapipe as mp
import numpy as np
import cv2 as cv
import time

def openCamera(width=1280, height=720, fps=30, camera_index=0):
    # Use OpenCV’s VideoCapture to start capturing from the webcam.
    # camera_index matters: adding a virtual camera device (like OBS's)
    # can shift which index your physical webcam ends up at, so this is
    # NOT guaranteed to always be 0. Use probe_cameras.py to find the
    # right value if the wrong device gets opened.
    cap = cv.VideoCapture(camera_index)

    # Request a moderate resolution/fps instead of the camera's native
    # (often much higher) default. Face-landmark detection doesn't need
    # 1080p/60fps, and requesting less here reduces load throughout.
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv.CAP_PROP_FPS, fps)

    # Warm-up: on some systems (notably macOS), the camera reports
    # isOpened() == True before the sensor is actually ready to deliver
    # frames, so the very first read() call(s) can fail even though
    # nothing is actually wrong. Retry briefly here so callers don't see
    # a spurious "stream end" on frame 1.
    if cap.isOpened():
        for _ in range(30):  # up to ~1.5s at 0.05s intervals
            ret, _ = cap.read()
            if ret:
                break
            time.sleep(0.05)

    return cap

def imgLoader(cap, retries=30, retry_delay=0.05):
    if cap is None or not cap.isOpened():
        print("Cannot open camera")
        return None, 0

    for attempt in range(retries):
        ret, frame = cap.read()
        if ret:
            return frame, ret
        time.sleep(retry_delay)

    # Only treat it as a real stream end after several consecutive
    # failed reads, not just one.
    print("Can't receive frame (stream end?). Exiting ...")
    return None, 0