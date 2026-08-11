# Face Filter

A real-time computer vision application that detects facial landmarks from a webcam and applies customizable AR-style face filters. The processed video can also be streamed through a **virtual camera** for use in applications such as Zoom, Google Meet, and Microsoft Teams.

Built with **Python, OpenCV, MediaPipe, NumPy, multiprocessing, shared memory, and pyvirtualcam**.

---

## Features

* Real-time face landmark detection using **MediaPipe Face Landmarker**
* Live webcam processing with **OpenCV**
* Custom PNG-based face filters with transparency
* Automatic filter **scaling, positioning, and rotation** based on facial landmarks
* Multiple built-in filters:

  * Spider-Man
  * Iron Man
  * Captain America
* Configurable camera and filter parameters through YAML
* Multiprocessing with **shared memory** for efficient frame sharing
* Processed output through a **virtual camera**
* Local preview for debugging and development

---

## Architecture

```text
             Webcam
                │
                ▼
        OpenCV Frame Capture
                │
                ▼
      MediaPipe Face Landmarker
                │
                ▼
       Facial Landmark Data
                │
                ▼
       Filter Processing
    (Scale / Rotate / Position)
                │
                ▼
         Processed Frame
                │
                ▼
          Shared Memory
                │
                ▼
         Virtual Camera
                │
                ▼
       Zoom / Meet / Teams
```

The main process handles webcam capture, face detection, and filter processing, while a separate process reads the latest frame from shared memory and sends it to the virtual camera.

---

## Project Structure

```text
Face-filter/
│
├── faceDetectionTracking/
│   ├── main.py              # Application entry point
│   ├── capture_worker.py    # Camera & face tracking pipeline
│   ├── maskOverlay.py       # Face filter processing
│   ├── imgAnnotation.py     # Landmark visualization
│   ├── imageLoader.py       # Camera handling
│   ├── vcam_worker.py       # Virtual camera output
│   ├── config.yaml          # Application configuration
│   │
│   ├── assets/masks/
│   │   ├── CA.png
│   │   ├── Ironman.png
│   │   ├── Spiderman.png
│   │   └── config_values.md
│   │
│   └── tests/
│       └── probe_cameras.py
│
└── README.md
```

---

## Tech Stack

* **Python**
* **OpenCV** – webcam capture and image processing
* **MediaPipe** – facial landmark detection
* **NumPy** – numerical and image operations
* **PyYAML** – configuration management
* **Multiprocessing & Shared Memory** – concurrent frame processing
* **pyvirtualcam** – virtual camera output

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Akarsh-09/Face-filter.git
cd Face-filter
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install numpy opencv-python mediapipe pyyaml pyvirtualcam
```

### 4. Add the MediaPipe model

Download a compatible **Face Landmarker** model and place it at:

```text
faceDetectionTracking/model/face_landmarker.task
```

### 5. Configure the application

Edit:

```text
faceDetectionTracking/config.yaml
```

Example:

```yaml
camera:
  index: 1
  width: 1920
  height: 1080
  fps: 60

display:
  show_local_preview: true

model:
  path: model/face_landmarker.task

mask:
  enabled: true
  image_path: assets/masks/CA.png
  top_padding_ratio: 0.55
  bottom_padding_ratio: 0.10
  horizontal_padding_ratio: 0.90
  vertical_offset_ratio: 0.0
```

### 6. Run

```bash
cd faceDetectionTracking
python main.py
```

---

## Adding a Custom Filter

Place a transparent PNG inside:

```text
faceDetectionTracking/assets/masks/
```

Then update `image_path` in `config.yaml`:

```yaml
mask:
  enabled: true
  image_path: assets/masks/my_filter.png
```

The padding and offset parameters can be adjusted to align the filter with the face.

---

## Requirements

* Python
* Webcam
* Compatible MediaPipe Face Landmarker model
* macOS recommended for the current implementation
* Virtual camera support for using the processed stream in other applications
