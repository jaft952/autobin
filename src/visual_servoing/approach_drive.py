"""
src/visual_servoing/approach_drive.py

Target error -> WheelCommand, via DifferentialKinematics. Pure function: no
hardware access here (Rule: hardware access only through src/hardware).
"""
from __future__ import annotations

from typing import Optional

from src.visual_servoing.distance_error import TargetError, STOP_DISTANCE_CM
from src.motion.differential_kinematics import DifferentialKinematics, WheelCommand

MIN_SPEED = 45.0            # TODO tune: below this the motors tend to stall
MAX_SPEED = 90.0             # TODO tune: matches MotionCalibration's default arc_speed
MAX_STEER_ANGLE_DEG = 75.0   # keep below 90 so it never fully spins in place
FAR_DISTANCE_CM = 80.0       # distance at/beyond which speed ramps to MAX_SPEED

# Below this distance, continuous driving risks overshoot: by the time a
# command reaches the wheels the frame it was computed from is already
# stale, and the can is close enough that a stale frame's worth of travel
# matters. is_final_approach() flags this zone; the caller (the live loop)
# is responsible for actually pulsing instead of driving continuously — see
# PULSE_* below, used there, not here (this module stays hardware/time-free).
PULSE_DISTANCE_CM = 35.0   # TODO tune: switch from cruise to pulse-then-look
PULSE_DURATION_S = 0.25    # TODO tune: length of one forward nudge
PULSE_PAUSE_S = 0.6        # TODO tune: stopped time for a fresh, unblurred look


def compute_drive_command(error: TargetError,
                           kin: DifferentialKinematics) -> Optional[WheelCommand]:
    """
    Returns None when there's nothing to drive toward: no target found, the
    target has been reached (arm handoff point), or it's flagged too_close
    (calibration-independent failsafe) — caller should stop in all cases.
    """
    if not error.found or error.reached or error.too_close:
        return None

    steer_angle = min(MAX_STEER_ANGLE_DEG,
                       abs(error.lateral_error) * 2 * MAX_STEER_ANGLE_DEG)

    if error.distance_cm is None:
        dynamic_speed = MIN_SPEED
    else:
        span = max(FAR_DISTANCE_CM - STOP_DISTANCE_CM, 1e-6)
        frac = (error.distance_cm - STOP_DISTANCE_CM) / span
        frac = max(0.0, min(1.0, frac))
        dynamic_speed = MIN_SPEED + frac * (MAX_SPEED - MIN_SPEED)

    # Sign convention matches src/subsumption/layers/layer2_approach.py:
    # lateral_error > 0 means the can is right of frame center -> steer right.
    if error.lateral_error > 0:
        return kin.arc_forward_right(angle_deg=steer_angle, speed=dynamic_speed)
    else:
        return kin.arc_forward_left(angle_deg=steer_angle, speed=dynamic_speed)


def is_final_approach(error: TargetError) -> bool:
    """True once close enough that continuous driving should give way to
    short pulse-then-look bursts instead (see PULSE_* above and the live
    loop that actually times them). False once already reached (nothing
    left to approach) or still far enough out that continuous cruising is
    fine and faster."""
    return (error.found and not error.reached and not error.too_close
            and error.distance_cm is not None
            and error.distance_cm <= PULSE_DISTANCE_CM)
