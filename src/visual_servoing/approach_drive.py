"""
src/visual_servoing/approach_drive.py

Target error -> WheelCommand, via DifferentialKinematics. Pure function: no
hardware access here (Rule: hardware access only through src/hardware).
"""
from __future__ import annotations

from typing import Optional

from src.visual_servoing.distance_error import TargetError, STOP_DISTANCE_CM
from src.motion.differential_kinematics import DifferentialKinematics, WheelCommand

MAX_SPEED = 30.0             # TODO tune: matches MotionCalibration's default arc_speed
FALLBACK_SPEED = 20.0        # TODO tune: used only when distance can't be estimated at all
BACKUP_SPEED = 30.0          # TODO tune: gentle reverse to recover from an overshoot
MAX_STEER_ANGLE_DEG = 20.0   # keep below 90 so it never fully spins in place
FAR_DISTANCE_CM = 30.0       # distance at/beyond which speed is MAX_SPEED


def compute_drive_command(error: TargetError,
                           kin: DifferentialKinematics) -> Optional[WheelCommand]:
    """
    Speed is the whole mechanism — no separate pulse/cruise mode. Between
    FAR_DISTANCE_CM and STOP_DISTANCE_CM the commanded speed ramps linearly
    from MAX_SPEED down to 0, so the approach is brisk far away and slows to
    a crawl exactly as it reaches the grab position, instead of stopping and
    restarting in bursts.

    Known limitation: because forward speed and steering both scale off the
    SAME ramped value (see DifferentialKinematics.arc_forward_*), a target
    that reaches STOP_DISTANCE_CM while still off-center enough that
    error.reached is False will get a zero-speed command and stop steering
    too — nothing pulls it the rest of the way to centered. Untested how
    often this actually happens in practice; if it does, the fix is a
    small in-place turn_left/turn_right correction for that specific case
    rather than an arc, but that's speculative until it's observed on
    hardware.

    Returns:
        None              — no target found, or reached (arm handoff point;
                            caller should stop).
        backward command  — too_close (bbox fills the frame): back off a
                            little regardless of the (possibly miscalibrated)
                            distance_cm math, so a bad calibration can't wedge
                            the robot against the can. Once backing off clears
                            too_close, the ramp below re-approaches and
                            re-centers on its own — no separate "recovery
                            state" needed.
        forward/arc command — otherwise, steered and speed-ramped toward the can.
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
        dynamic_speed = FALLBACK_SPEED
    else:
        span = max(FAR_DISTANCE_CM - STOP_DISTANCE_CM, 1e-6)
        frac = (error.distance_cm - STOP_DISTANCE_CM) / span
        frac = max(0.0, min(1.0, frac))
        dynamic_speed = MAX_SPEED * frac   # -> 0 right at STOP_DISTANCE_CM

    # lateral_error > 0 means the can is right of frame center -> steer right
    # toward it. Hardware-verified 2026-07-22 on this robot: that maps to
    # arc_forward_LEFT, not arc_forward_right — swapped from the naive
    # reading of layer2_approach.py's sign convention (that module never
    # actually calls these two kin methods; it builds its own WheelCommand
    # via MotionExecutor's v_x/v_theta mix, which is a different code path).
    if error.lateral_error > 0:
        return kin.arc_forward_left(angle_deg=steer_angle, speed=dynamic_speed)
    else:
        return kin.arc_forward_right(angle_deg=steer_angle, speed=dynamic_speed)
