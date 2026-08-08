import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import os
import yaml
import capture_worker
import vcam_worker

current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, 'config.yaml')

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

WIDTH = config['camera']['width']
HEIGHT = config['camera']['height']
FPS = config['camera']['fps']
CAMERA_INDEX = config['camera']['index']
SHOW_LOCAL_PREVIEW = config['display']['show_local_preview']
MODEL_PATH = os.path.join(current_dir, config['model']['path'])

if __name__ == "__main__":
    frame_shape = (HEIGHT, WIDTH, 3)
    frame_nbytes = int(np.prod(frame_shape))

    # Shared memory block both sides attach to. The main process (capture)
    # writes the latest annotated frame here; the vcam child process reads
    # from it.
    shm = shared_memory.SharedMemory(create=True, size=frame_nbytes)

    # Incremented every time a new frame is written, so the vcam process
    # can tell when there's something new to send instead of resending a
    # stale frame.
    frame_id = mp.Value('l', 0)

    # Shared shutdown signal -- either side can set this and both exit.
    stop_event = mp.Event()

    # NOTE: capture_worker (camera + MediaPipe) intentionally runs directly
    # in THIS process, not spawned. Testing showed camera access breaks
    # inside a multiprocessing-spawned child on this system, but works
    # reliably in the main process. Only the virtual camera sender -- which
    # doesn't need camera permissions -- is spawned as a child process.
    vcam_proc = mp.Process(
        target=vcam_worker.run,
        args=(shm.name, frame_shape, frame_id, stop_event, WIDTH, HEIGHT, FPS),
    )
    vcam_proc.start()

    try:
        capture_worker.run(
            shm.name, frame_shape, frame_id, stop_event,
            WIDTH, HEIGHT, FPS, CAMERA_INDEX,
            model_path=MODEL_PATH, show_local_preview=SHOW_LOCAL_PREVIEW,
        )
    except KeyboardInterrupt:
        stop_event.set()

    stop_event.set()
    vcam_proc.join(timeout=5)
    if vcam_proc.is_alive():
        vcam_proc.terminate()

    shm.close()
    shm.unlink()

    print("Shut down cleanly.")