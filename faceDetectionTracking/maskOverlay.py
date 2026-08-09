import cv2
import numpy as np

LEFT_EYE_OUTER = 263
RIGHT_EYE_OUTER = 33
FOREHEAD_TOP = 10
CHIN_BOTTOM = 152


def load_mask(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not load mask image at: {path}")

    if img.shape[2] == 3:
        alpha = np.full(img.shape[:2], 255, dtype=np.uint8)
        img = np.dstack([img, alpha])

    return img


def _landmarks_to_pixels(face_landmarks, frame_width, frame_height):
    """Converts MediaPipe's normalized [0,1] landmarks to pixel coords."""
    return np.array(
        [[lm.x * frame_width, lm.y * frame_height] for lm in face_landmarks],
        dtype=np.float32,
    )


def apply_mask_overlay(bgr_frame, detection_result, mask_rgba,
                        top_padding_ratio, bottom_padding_ratio,
                        horizontal_padding_ratio, vertical_offset_ratio):

    if not detection_result.face_landmarks:
        return bgr_frame
 
    face_landmarks = detection_result.face_landmarks[0]
    h, w = bgr_frame.shape[:2]
    points = _landmarks_to_pixels(face_landmarks, w, h)
 
    # --- Bounding box from landmark extremes ---
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    box_w = x_max - x_min
    box_h = y_max - y_min
 
    # --- Extend the box asymmetrically ---
    ext_y_min = y_min - top_padding_ratio * box_h
    ext_y_max = y_max + bottom_padding_ratio * box_h
    ext_x_min = x_min - horizontal_padding_ratio * box_w
    ext_x_max = x_max + horizontal_padding_ratio * box_w
 
    cx = (ext_x_min + ext_x_max) / 2.0
    cy = (ext_y_min + ext_y_max) / 2.0
 
    # --- Rotation angle from the eye line ---
    left_eye = points[LEFT_EYE_OUTER]
    right_eye = points[RIGHT_EYE_OUTER]
    dx, dy = (left_eye - right_eye)
    angle_deg = np.degrees(np.arctan2(dy, dx))
 
    # --- Target size for the mask, from the extended box ---
    target_w = max(1, int(ext_x_max - ext_x_min))
    target_h = max(1, int(ext_y_max - ext_y_min))
 
    resized_mask = cv2.resize(mask_rgba, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
 
    # --- Rotate the resized mask around its own center ---
    rot_center = (target_w / 2.0, target_h / 2.0)
    rot_mat = cv2.getRotationMatrix2D(rot_center, -angle_deg, 1.0)
    rotated_mask = cv2.warpAffine(
        resized_mask, rot_mat, (target_w, target_h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
    )
 
    # --- Compute placement on the frame ---
    top_left_x = int(cx - target_w / 2.0)
    top_left_y = int(cy - target_h / 2.0 + vertical_offset_ratio * target_h)
 
    # Clip to frame bounds (mask may partially go offscreen near edges)
    src_x0, src_y0 = 0, 0
    dst_x0, dst_y0 = top_left_x, top_left_y
    dst_x1, dst_y1 = top_left_x + target_w, top_left_y + target_h
 
    if dst_x0 < 0:
        src_x0 = -dst_x0
        dst_x0 = 0
    if dst_y0 < 0:
        src_y0 = -dst_y0
        dst_y0 = 0
    dst_x1 = min(dst_x1, w)
    dst_y1 = min(dst_y1, h)
 
    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return bgr_frame
 
    region_w = dst_x1 - dst_x0
    region_h = dst_y1 - dst_y0
    mask_crop = rotated_mask[src_y0:src_y0 + region_h, src_x0:src_x0 + region_w]
 
    mask_rgb = mask_crop[:, :, :3].astype(np.float32)
    mask_alpha = (mask_crop[:, :, 3:4].astype(np.float32)) / 255.0
 
    output = bgr_frame.copy()
    frame_region = output[dst_y0:dst_y1, dst_x0:dst_x1].astype(np.float32)
 
    blended = mask_alpha * mask_rgb + (1.0 - mask_alpha) * frame_region
    output[dst_y0:dst_y1, dst_x0:dst_x1] = blended.astype(np.uint8)
 
    return output
 