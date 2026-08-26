"""
src/visual_servoing/distance_error.py

Perception -> target error. Pure function: takes a DetectionResult (see
src/perception/detector.py), never a live camera/sensor object, so this is
unit-testable with synthetic fixtures and needs no hardware.

Lateral error and distance both anchor on normalized_base_center() (the
ground-contact point) rather than the bbox center, matching the convention
arc-grasp and pixel-to-arm already use for the same can.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

from src.perception.detector import DetectionResult

# calibration
CALIBRATION_CONSTANT_PX_CM = 48258720

STOP_DISTANCE_CM = 25.0
CENTER_TOLERANCE = 0.06

CLOSE_BBOX_FRACTION = 0.75 # TODO tune: fraction of frame_height
TOO_CLOSE_DISTANCE_CM = 10.0


@dataclass
class TargetError:
    found: bool
    lateral_error: float          # -0.5..0.5, +ve = can right of frame center
    distance_cm: Optional[float]  # None if not found / not estimable
    reached: bool                 # within STOP_DISTANCE_CM and centered
    too_close: bool = False       # bbox fills the frame -> stop regardless of calibration


def compute_target_error(detection: DetectionResult,
                          ultrasonic_cm: Optional[float] = None) -> TargetError:
    """
    ultrasonic_cm: reserved for a future hybrid vision+ultrasonic distance
    fusion (no ultrasonic sensor is fitted yet — always pass None today).
    When it's added later, blend/override the monocular estimate near the
    end of this function; the signature is already future-proofed for it.
    """
    base = detection.normalized_base_center() # ground-contact point
    if base is None or not detection.found:
        return TargetError(found=False, lateral_error=0.0,
                            distance_cm=None, reached=False, too_close=False)

    x, _y = base
    lateral_error = x - 0.5

    bbox_height_px = detection.best.height # type: ignore

    # Prefer the actual segmented pixel count over the rectangular bbox --
    # a bbox includes background around a round/angled can and overstates
    # its footprint. Falls back to bbox area if no mask is available (e.g.
    # the seg model didn't return one for this detection).
    area_px = detection.best.mask_area or (detection.best.width * bbox_height_px) # type: ignore

    distance_cm = (
        math.sqrt(CALIBRATION_CONSTANT_PX_CM / area_px)
        if area_px > 0 else None
    )

    too_close = (detection.frame_height > 0
                 and bbox_height_px / detection.frame_height >= CLOSE_BBOX_FRACTION)

    if distance_cm is not None:
        too_close = too_close or (distance_cm <= TOO_CLOSE_DISTANCE_CM)

    if ultrasonic_cm is not None and distance_cm is not None:
        too_close = too_close or (ultrasonic_cm <= STOP_DISTANCE_CM)

    reached = (distance_cm is not None
               and distance_cm <= STOP_DISTANCE_CM
               and abs(lateral_error) <= CENTER_TOLERANCE)

    return TargetError(found=True, lateral_error=lateral_error,
                        distance_cm=distance_cm, reached=reached, too_close=too_close)
