"""
MediaPipe Face Mesh (legacy solutions API) landmark detection.
Uses mp.solutions.face_mesh with refine_landmarks=True to get
iris landmarks (468-477) in addition to the 468-point face mesh.

Iris centers, iris radius (→ px/mm via 11.7mm), canthal points,
and upper eyelid margin points for both eyes.
"""

import os
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional

import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

# ── Iris centers ─────────────────────────────────────────────────────────────
LEFT_IRIS_CENTER       = 473
RIGHT_IRIS_CENTER      = 468
RIGHT_IRIS_ALL4        = [469, 470, 471, 472]
LEFT_IRIS_ALL4         = [474, 475, 476, 477]

# ── Canthal points ───────────────────────────────────────────────────────────
LEFT_MEDIAL_CANTHUS    = 362
LEFT_LATERAL_CANTHUS   = 263
RIGHT_MEDIAL_CANTHUS   = 133
RIGHT_LATERAL_CANTHUS  = 33

# ── Upper eyelid MARGIN landmarks (lateral → medial) ─────────────────────────
RIGHT_UPPER_LID_MARGIN = [246, 161, 160, 159, 158, 157, 173]
LEFT_UPPER_LID_MARGIN  = [466, 388, 387, 386, 385, 384, 398]

IRIS_DIAMETER_MM = 11.7


@dataclass
class EyeLandmarks:
    pupil_center:       np.ndarray   # (x, y) px
    iris_radius_px:     float
    px_per_mm:          float
    medial_canthus:     np.ndarray
    lateral_canthus:    np.ndarray
    upper_lid_margin:   np.ndarray   # (N, 2) ordered lateral→medial


@dataclass
class LandmarkResult:
    left_eye:       Optional[EyeLandmarks] = None
    right_eye:      Optional[EyeLandmarks] = None
    confidence:     float = 0.0
    low_confidence: bool  = False
    warning_msg:    str   = ""
    image_rgb:      Optional[np.ndarray] = None


def detect_landmarks(image_bgr: np.ndarray) -> LandmarkResult:
    result    = LandmarkResult()
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result.image_rgb = image_rgb
    h, w = image_bgr.shape[:2]

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,      # enables iris landmarks 468-477
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        detection = face_mesh.process(image_rgb)

    if not detection.multi_face_landmarks:
        result.warning_msg    = "No face detected. Please use a clear frontal face photo."
        result.low_confidence = True
        return result

    lms = detection.multi_face_landmarks[0].landmark

    def lm(idx) -> np.ndarray:
        pt = lms[idx]
        return np.array([pt.x * w, pt.y * h], dtype=float)

    def iris_radius_robust(center_idx: int, all4: list) -> float:
        c = lm(center_idx)
        radii = [float(np.linalg.norm(lm(b) - c)) for b in all4]
        return float(np.mean(radii))

    def is_eye_open(center_idx: int, radius_px: float) -> bool:
        cx, cy = lm(center_idx)
        r = max(3, int(radius_px * 0.7))
        x1 = max(0, int(cx) - r);  x2 = min(w, int(cx) + r)
        y1 = max(0, int(cy) - r);  y2 = min(h, int(cy) + r)
        roi = cv2.cvtColor(image_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        if roi.size == 0:
            return False
        return float(np.mean(roi)) < 95.0

    def light_reflex(center_idx: int, radius_px: float) -> np.ndarray:
        cx, cy = lm(center_idx)
        r = int(radius_px * 0.6)
        x1 = max(0, int(cx) - r);  x2 = min(w, int(cx) + r)
        y1 = max(0, int(cy) - r);  y2 = min(h, int(cy) + r)
        roi = cv2.cvtColor(image_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        if roi.size == 0:
            return np.array([cx, cy], dtype=float)
        if float(np.mean(roi)) > 90:
            return np.array([cx, cy], dtype=float)
        blurred = cv2.GaussianBlur(roi.astype(np.float32), (5, 5), 0)
        idx = np.unravel_index(np.argmax(blurred), blurred.shape)
        return np.array([x1 + idx[1], y1 + idx[0]], dtype=float)

    # Right eye
    r_radius = iris_radius_robust(RIGHT_IRIS_CENTER, RIGHT_IRIS_ALL4)
    r_ppmm   = (r_radius * 2.0) / IRIS_DIAMETER_MM if r_radius > 0 else 1.0
    if is_eye_open(RIGHT_IRIS_CENTER, r_radius):
        r_reflex = light_reflex(RIGHT_IRIS_CENTER, r_radius)
        result.right_eye = EyeLandmarks(
            pupil_center    = r_reflex,
            iris_radius_px  = r_radius,
            px_per_mm       = r_ppmm,
            medial_canthus  = lm(RIGHT_MEDIAL_CANTHUS),
            lateral_canthus = lm(RIGHT_LATERAL_CANTHUS),
            upper_lid_margin= np.array([lm(i) for i in RIGHT_UPPER_LID_MARGIN]),
        )

    # Left eye
    l_radius = iris_radius_robust(LEFT_IRIS_CENTER, LEFT_IRIS_ALL4)
    l_ppmm   = (l_radius * 2.0) / IRIS_DIAMETER_MM if l_radius > 0 else 1.0
    if is_eye_open(LEFT_IRIS_CENTER, l_radius):
        l_reflex = light_reflex(LEFT_IRIS_CENTER, l_radius)
        result.left_eye = EyeLandmarks(
            pupil_center    = l_reflex,
            iris_radius_px  = l_radius,
            px_per_mm       = l_ppmm,
            medial_canthus  = lm(LEFT_MEDIAL_CANTHUS),
            lateral_canthus = lm(LEFT_LATERAL_CANTHUS),
            upper_lid_margin= np.array([lm(i) for i in LEFT_UPPER_LID_MARGIN]),
        )

    min_radius        = min(r_radius, l_radius)
    confidence        = min(1.0, min_radius / (min(h, w) * 0.02))
    result.confidence     = confidence
    result.low_confidence = confidence < 0.5
    if result.low_confidence:
        result.warning_msg = (
            f"Low landmark confidence ({confidence:.2f}). "
            "Use a well-lit frontal photo."
        )
    return result
