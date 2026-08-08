import cv2 as cv
import time

def openCamera(width=1280, height=720, fps=30, camera_index=1):
    cap = cv.VideoCapture(camera_index)

    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv.CAP_PROP_FPS, fps)

    if cap.isOpened():
        for _ in range(30):
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

    print("Can't receive frame (stream end?). Exiting ...")
    return None, 0