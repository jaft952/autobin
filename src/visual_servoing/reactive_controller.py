"""
src/visual_servoing/reactive_controller.py

Target error -> WheelCommand, via DifferentialKinematics. Pure function: no
hardware access here (Rule: hardware access only through src/hardware).
"""
from __future__ import annotations

from typing import Optional

from src.visual_servoing.distance_error import TargetError
from src.motion.differential_kinematics import DifferentialKinematics, WheelCommand

FAR_DISTANCE_CM = 100.0          # distance at/below which speed drops from FORWARD_HIGH_SPEED
LOW_DISTANCE_CM = 50.0          # distance at/below which speed is LOW_SPEED

FORWARD_HIGH_SPEED = 32.0       # hand-tested cruise speed at/beyond FAR_DISTANCE
FORWARD_MID_SPEED = 30.0        # hand-tested speed between FAR_DISTANCE_CM and MID_DISTANCE_CM
FORWARD_LOW_SPEED = 28.0        # used only when distance can't be estimated at all

BACKUP_SPEED = 15.0             # gentle reverse to recover from an overshoot
MAX_STEER_ANGLE_DEG = 90.0      # keep below 90 so it never fully spins in place


def compute_reactive_command(error: TargetError,
                           kin: DifferentialKinematics) -> Optional[WheelCommand]:
    """
    Speed is a lookup against hardcoded distance breakpoints, not a
    continuous formula (see the constants block above for why):
        distance_cm >= FAR_DISTANCE_CM  (~100cm) -> MAX_SPEED
        FAR_DISTANCE_CM < distance_cm < MID_DISTANCE_CM (~50cm) -> MID_SPEED
        distance_cm <= LOW_DISTANCE_CM     (~50cm) -> LOW_SPEED

    That last tier never reaches zero -- unlike the old zero-at-STOP_DISTANCE_CM
    ramp this replaced, steering keeps working the whole way to the grab
    point since dynamic_speed (which both forward speed AND steer angle scale
    off, via DifferentialKinematics.arc_forward_*) never collapses to
    nothing.

    Returns:
        None              — no target found, or reached (arm handoff point;
                            caller should stop).
        backward command  — too_close (bbox fills the frame): back off a
                            little regardless of the (possibly miscalibrated)
                            distance_cm math, so a bad calibration can't wedge
                            the robot against the can. Once backing off clears
                            too_close, the tiers above re-approach and
                            re-center on their own — no separate "recovery
                            state" needed.
        forward/arc command — otherwise, steered and speed-tiered toward the can.
    """
    if not error.found:
        return None

    if error.too_close:
        return kin.backward(speed=BACKUP_SPEED)

    if error.reached:
        return None

    steer_angle = min(MAX_STEER_ANGLE_DEG,
                       abs(error.lateral_error) * 2 * MAX_STEER_ANGLE_DEG)

    if error.distance_cm is None:
        dynamic_speed = 0
    elif error.distance_cm > FAR_DISTANCE_CM:
        dynamic_speed = FORWARD_HIGH_SPEED
    elif error.distance_cm > LOW_DISTANCE_CM and error.distance_cm <= FAR_DISTANCE_CM:
        dynamic_speed = FORWARD_MID_SPEED
    elif error.distance_cm > 0 and error.distance_cm <= LOW_DISTANCE_CM:
        dynamic_speed = FORWARD_LOW_SPEED
    else:
        dynamic_speed = 0

    if error.lateral_error > 0:
        return kin.arc_forward_left(angle_deg=steer_angle, speed=dynamic_speed)
    else:
        return kin.arc_forward_right(angle_deg=steer_angle, speed=dynamic_speed)


def speed_tier(error: TargetError) -> str:
    """Which branch of compute_reactive_command's logic this error would hit,
    as a short label -- for logging/display only (e.g. the live loop's
    console/overlay monitor). Mirrors that function's branching exactly, so
    the readout can't silently drift out of sync with the actual thresholds
    as they get re-tuned; keep the two in sync if the branches ever change.

    One of: "none" (not found), "reached", "backup" (too_close), "fallback"
    (distance_cm unknown), "cruise", "approach", "step".
    """
    if not error.found:
        return "none"
    if error.too_close:
        return "backup"
    if error.reached:
        return "reached"
    if error.distance_cm is None:
        return "fallback"
    if error.distance_cm > FAR_DISTANCE_CM:
        return "cruise"
    if error.distance_cm > LOW_DISTANCE_CM and error.distance_cm <= FAR_DISTANCE_CM:
        return "approach"
    if error.distance_cm > 0 and error.distance_cm <= LOW_DISTANCE_CM:
        return "step"
    return "fallback"