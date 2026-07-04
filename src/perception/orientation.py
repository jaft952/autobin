"""
src/perception/orientation.py

Orientation estimate for a detected can. When the detection carries a real
segmentation mask (YOLO11-seg -> BoundingBox.mask_poly), the rotated rectangle
is fitted directly to that mask outline — no thresholding, robust to busy
backgrounds. Boxes without a mask (detect-only models, synthetic test boxes)
fall back to the classical-CV path: segment the can inside the bbox crop and
fit cv2.minAreaRect to the largest contour.

    angle  — the can's long-axis angle in the image (deg, 0..180; ~90 = vertical)
    aspect — long/short side ratio (>= 1)
    klass  — "upright" | "lying" | "axial"

Used by the grasp pipeline to pick the grasp target point and the gripper roll:
    upright  -> top-down grasp, base point, roll doesn't matter (symmetry)
    lying    -> grasp the can CENTER, roll CH5 to (angle + 90)
    axial    -> looks round => axis points ~toward the robot => fixed roll

cv2 / numpy are imported lazily inside the functions so importing this module
(e.g. the Orientation dataclass) never loads a native library.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Orientation:
    angle: float    # long-axis angle in the image (deg, 0..180; ~90 = vertical)
    aspect: float   # long/short side ratio (>= 1.0)
    klass: str      # "upright" | "lying" | "axial"


def _classical_contour(frame_bgr, box):
    """Fallback when no real mask is available: segment the can inside its
    bbox crop and return the largest contour (in crop coords), or None.
    Angle/aspect are translation-invariant, so crop coords are fine."""
    import cv2
    import numpy as np

    x1, y1 = max(0, int(box.x1)), max(0, int(box.y1))
    x2, y2 = int(box.x2), int(box.y2)
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0 or min(crop.shape[:2]) < 8:
        return None

    # Foreground mask: colourful cans pop on the saturation channel; fall back to
    # a grayscale Otsu split for low-saturation (plain metal) cans.
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    _, mask = cv2.threshold(hsv[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if cv2.countNonZero(mask) < 0.15 * mask.size:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if cv2.countNonZero(mask) > 0.5 * mask.size:
            mask = cv2.bitwise_not(mask)  # keep the can (smaller blob) as foreground

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 0.05 * mask.size:
        return None
    return cnt


def estimate_orientation(frame_bgr, box, round_aspect: float = 1.35,
                         upright_band_deg: float = 35.0):
    """Estimate a can's orientation. Prefers the real YOLO11-seg mask outline
    (box.mask_poly, Nx2 pixel coords); otherwise segments the bbox crop with
    classical CV. Returns an Orientation, or None if the can couldn't be
    segmented.

    `box` only needs .x1/.y1/.x2/.y2 (+ optional .mask_poly) — duck-typed."""
    import cv2
    import numpy as np

    poly = getattr(box, "mask_poly", None)
    if poly is not None and len(poly) >= 3:
        cnt = np.asarray(poly, dtype=np.float32).reshape(-1, 1, 2)
    else:
        cnt = _classical_contour(frame_bgr, box)
        if cnt is None:
            return None

    (_, (w, h), _) = cv2.minAreaRect(cnt)
    if w == 0 or h == 0:
        return None
    pts = cv2.boxPoints(cv2.minAreaRect(cnt))

    # Long-axis angle = direction of the rect's longest edge (version-independent).
    edges = [(pts[i], pts[(i + 1) % 4]) for i in range(4)]
    p1, p2 = max(edges, key=lambda e: float(np.hypot(*(e[1] - e[0]))))
    angle = float(np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0])) % 180.0)
    aspect = float(max(w, h) / min(w, h))

    if aspect < round_aspect:
        klass = "axial"                              # round -> axis toward robot
    elif abs(angle - 90.0) <= upright_band_deg:
        klass = "upright"                            # long axis ~vertical
    else:
        klass = "lying"                              # long axis horizontal/diagonal

    return Orientation(angle=round(angle, 1), aspect=round(aspect, 2), klass=klass)
