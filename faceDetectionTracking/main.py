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
MASK_CONFIG = config.get('mask', {'enabled': False})

if __name__ == "__main__":
    frame_shape = (HEIGHT, WIDTH, 3)
    frame_nbytes = int(np.prod(frame_shape))

    # Shared memory block both sides attach to. The main process (capture)
    # writes the latest annotated frame here; the vcam child process reads
    # from it.
    shm = shared_memory.SharedMemory(create=True, size=frame_nbytes)

    frame_id = mp.Value('l', 0)
    stop_event = mp.Event()

    # NOTE: capture_worker (camera + MediaPipe) runs directly in THIS
    # process, not spawned.
    # Only the virtual camera sender -- spawned as a child process.
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
            mask_config=MASK_CONFIG,
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