import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import imageLoader as imgLoader
import imgAnnotation as imgAn
import time
import cv2
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'model', 'face_landmarker.task')

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

stop = False
frame_buffer = {}
latest_frame = None
detection_busy = False

def print_result(result, image, timestamp_ms):
    global latest_frame, detection_busy

    rgb_frame = frame_buffer.pop(timestamp_ms, None)
    if rgb_frame is None:
        detection_busy = False
        return

    annotated_image = imgAn.draw_landmarks_on_image(rgb_frame, result)
    latest_frame = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    detection_busy = False

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path, delegate=BaseOptions.Delegate.CPU),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

with FaceLandmarker.create_from_options(options) as landmarker:
    cap = imgLoader.openCamera()

    while not stop:
        frame, ret = imgLoader.imgLoader(cap)
        if ret == 0:
            break

        if not detection_busy:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_timestamp_ms = time.time_ns() // 1_000_000
            frame_buffer[frame_timestamp_ms] = rgb_frame
 
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_busy = True
            landmarker.detect_async(mp_image, frame_timestamp_ms)

        display_frame = latest_frame if latest_frame is not None else frame

        cv2.imshow('Face Landmarker', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop = True

    cap.release()
    cv2.destroyAllWindows()
