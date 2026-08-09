import mediapipe as mp
import imageLoader
import imgAnnotation
import maskOverlay
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
        camera_index, model_path, show_local_preview,
        mask_config):

    mask_enabled = bool(mask_config and mask_config.get("enabled"))
    mask_rgba = None
    if mask_enabled:
        mask_path = mask_config["image_path"]
        if not os.path.isabs(mask_path):
            mask_path = os.path.join(current_dir, mask_path)
        mask_rgba = maskOverlay.load_mask(mask_path)

    frame_buffer = {}
    state = {"latest_frame": None, "detection_busy": False}

    shm = shared_memory.SharedMemory(name=shm_name)
    shared_arr = np.ndarray(frame_shape, dtype=np.uint8, buffer=shm.buf)

    def print_result(result, image, timestamp_ms):
        rgb_frame = frame_buffer.pop(timestamp_ms, None)
        if rgb_frame is None:
            state["detection_busy"] = False
            return

        if mask_enabled:
            bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            state["latest_frame"] = maskOverlay.apply_mask_overlay(
                bgr_frame, result, mask_rgba,
                top_padding_ratio=mask_config.get("top_padding_ratio"),
                bottom_padding_ratio=mask_config.get("bottom_padding_ratio"),
                horizontal_padding_ratio=mask_config.get("horizontal_padding_ratio"),
                vertical_offset_ratio=mask_config.get("vertical_offset_ratio"),
            )
        else:
            annotated_image = imgAnnotation.draw_landmarks_on_image(rgb_frame, result)
            state["latest_frame"] = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

        state["detection_busy"] = False

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path, delegate=BaseOptions.Delegate.CPU),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=print_result)

    with FaceLandmarker.create_from_options(options) as landmarker:
        cap = imageLoader.openCamera(width=width, height=height, fps=fps, camera_index=camera_index)

        if cap is None or not cap.isOpened():
            print("Could not access the physical camera.")
            stop_event.set()
            shm.close()
            return

        while not stop_event.is_set():
            frame, ret = imageLoader.imgLoader(cap)
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