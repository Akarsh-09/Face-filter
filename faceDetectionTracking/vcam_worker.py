import numpy as np
from multiprocessing import shared_memory
import pyvirtualcam
from pyvirtualcam import PixelFormat


def run(shm_name, frame_shape, frame_id, stop_event, width, height, fps):

    shm = shared_memory.SharedMemory(name=shm_name)
    shared_arr = np.ndarray(frame_shape, dtype=np.uint8, buffer=shm.buf)

    # print(f"SHM name while virtualizing: {shm.name}")

    with pyvirtualcam.Camera(width=width, height=height, fps=fps, fmt=PixelFormat.BGR) as cam:
        print(f"Virtual camera running: {cam.device}")
        print("In Zoom/Teams/Chrome, select this device as your camera input.")

        last_seen_id = -1
        while not stop_event.is_set():
            current_id = frame_id.value
            if current_id != last_seen_id:
                frame = shared_arr.copy()
                cam.send(frame)
                last_seen_id = current_id
            cam.sleep_until_next_frame()

    shm.close()