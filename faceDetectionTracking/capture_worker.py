import mediapipe as mp
import imageLoader as imgLoader
import imgAnnotation as imgAn
import time
import cv2
import os
from multiprocessing import shared_memory
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(current_dir, 'model', 'face_landmarker.task')

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def run(shm_name, frame_shape, frame_id, stop_event, width, height, fps,
        camera_index=0, model_path=None, show_local_preview=True):

    if model_path is None:
        model_path = DEFAULT_MODEL_PATH

    # print(f"SHM name while capturing: {shm_name}")

    frame_buffer = {}
    state = {"latest_frame": None, "detection_busy": False}

    shm = shared_memory.SharedMemory(name=shm_name)
    shared_arr = np.ndarray(frame_shape, dtype=np.uint8, buffer=shm.buf)

    def print_result(result, image, timestamp_ms):
        rgb_frame = frame_buffer.pop(timestamp_ms, None)
        if rgb_frame is None:
            state["detection_busy"] = False
            return

        annotated_image = imgAn.draw_landmarks_on_image(rgb_frame, result)
        state["latest_frame"] = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        state["detection_busy"] = False

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path, delegate=BaseOptions.Delegate.CPU),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=print_result)

    with FaceLandmarker.create_from_options(options) as landmarker:
        cap = imgLoader.openCamera(width=width, height=height, fps=fps, camera_index=camera_index)

        if cap is None or not cap.isOpened():
            print("Could not access the physical camera.")
            stop_event.set()
            shm.close()
            return

        while not stop_event.is_set():
            frame, ret = imgLoader.imgLoader(cap)
            if ret == 0:
                break

            if not state["detection_busy"]:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_timestamp_ms = time.time_ns() // 1_000_000
                frame_buffer[frame_timestamp_ms] = rgb_frame

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                state["detection_busy"] = True
                landmarker.detect_async(mp_image, frame_timestamp_ms)

            display_frame = state["latest_frame"] if state["latest_frame"] is not None else frame

            if display_frame.shape != frame_shape:
                display_frame = cv2.resize(display_frame, (width, height))

            shared_arr[:] = display_frame
            with frame_id.get_lock():
                frame_id.value += 1

            if show_local_preview:
                cv2.imshow('Face Landmarker (local preview)', display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_event.set()
                    break

        cap.release()
        cv2.destroyAllWindows()

    stop_event.set()
    shm.close()