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
from typing import Optional

from src.perception.detector import DetectionResult

# ── Calibration (measure on hardware) ───────────────────────────────────
CAN_HEIGHT_CM = 12.2  # TODO verify against the actual target tin can

# Monocular pinhole distance estimate: distance_cm = CONSTANT / bbox_height_px.
# Measure bbox_height_px for the can placed at a known distance_cm, then set:
#   CALIBRATION_CONSTANT_PX_CM = bbox_height_px * distance_cm
CALIBRATION_CONSTANT_PX_CM = 4000.0  # TODO calibrate on hardware

STOP_DISTANCE_CM = 15.0   # TODO tune: distance at which the arm can grasp
CENTER_TOLERANCE = 0.06   # normalized lateral error considered "centered"


@dataclass
class TargetError:
    found: bool
    lateral_error: float          # -0.5..0.5, +ve = can right of frame center
    distance_cm: Optional[float]  # None if not found / not estimable
    reached: bool                  # within STOP_DISTANCE_CM and centered


def compute_target_error(detection: DetectionResult,
                          ultrasonic_cm: Optional[float] = None) -> TargetError:
    """
    ultrasonic_cm: reserved for a future hybrid vision+ultrasonic distance
    fusion (no ultrasonic sensor is fitted yet — always pass None today).
    When it's added later, blend/override the monocular estimate near the
    end of this function; the signature is already future-proofed for it.
    """
    base = detection.normalized_base_center()   # ground-contact point
    if base is None or not detection.found:
        return TargetError(found=False, lateral_error=0.0,
                            distance_cm=None, reached=False)

    x, _y = base
    lateral_error = x - 0.5

    bbox_height_px = detection.best.height
    distance_cm = (CALIBRATION_CONSTANT_PX_CM / bbox_height_px
                    if bbox_height_px > 0 else None)

    # Extension seam for future ultrasonic fusion — no-op today.
    if ultrasonic_cm is not None and distance_cm is not None:
        pass  # TODO: fuse once an ultrasonic sensor is fitted

    reached = (distance_cm is not None
               and distance_cm <= STOP_DISTANCE_CM
               and abs(lateral_error) <= CENTER_TOLERANCE)

    return TargetError(found=True, lateral_error=lateral_error,
                        distance_cm=distance_cm, reached=reached)
