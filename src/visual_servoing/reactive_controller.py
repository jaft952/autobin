"""
src/visual_servoing/reactive_controller.py

Target error -> WheelCommand, via DifferentialKinematics. Pure function: no
hardware access here (Rule: hardware access only through src/hardware).

Also exposes is_final_approach(), a pure predicate the live loop uses to
decide WHEN to switch from driving compute_reactive_command's output
continuously (cruising) to applying it in short step-then-look bursts
timing itself (STEP_DURATION_S / LOOK_PAUSE_S) is the caller's job, same as
the rest of hardware/timing.
"""
from __future__ import annotations

from typing import Optional

from src.visual_servoing.distance_error import TargetError
from src.motion.differential_kinematics import DifferentialKinematics, WheelCommand

# Speed is looked up from these hand-tested distance breakpoints, not
# computed from a continuous formula -- distance_cm is only as good as
# distance_error.CALIBRATION_CONSTANT_PX_CM, which is still an unmeasured
# placeholder (see that module), so a smooth ramp off it is no more
# trustworthy than a few hardcoded checkpoints tuned by eye on hardware.
CRUISE_DISTANCE_CM = 90.0    # distance at/beyond which speed is MAX_SPEED
MAX_SPEED = 25.0             # TODO tune: hand-tested cruise speed at/beyond CRUISE_DISTANCE_CM
APPROACH_SPEED = 25.0        # TODO tune: hand-tested speed between FAR_DISTANCE_CM and CRUISE_DISTANCE_CM
FALLBACK_SPEED = 20.0        # TODO tune: used only when distance can't be estimated at all

# Lowered from 24 (same as MAX_SPEED) -- on hardware that drove the
# too_close backup at full speed for as long as too_close stayed True,
# which overshot well past a safe recovery distance before the next
# (stale-by-the-time-it-arrives) frame could clear the flag. Still just a
# guess pending another hardware pass -- watch for either "still backs up
# too far" (lower further) or "doesn't clear too_close fast enough" (raise
# it back up a bit).
BACKUP_SPEED = 20.0          # TODO tune: gentle reverse to recover from an overshoot

# Used by the live loop (not compute_reactive_command itself -- there's no
# TargetError case that means "scan"): when the ultrasonic stops the robot
# but vision doesn't confirm it's the tracked can (UltrasonicState.
# matches_vision is False), the caller turns in place at this speed instead
# of driving toward an unconfirmed obstacle, looking for a new target.
# Deliberately slower than BACKUP_SPEED -- this is a search, not a recovery,
# and a slow turn gives the camera a real chance to pick something up
# instead of sweeping past it.
SCAN_SPEED = 15.0            # TODO tune: slow in-place turn while scanning for a new can

MAX_STEER_ANGLE_DEG = 80.0   # keep below 90 so it never fully spins in place

# Distance control is handled by ultrasonic sensor.
# Vision is only responsible for horizontal alignment.
TARGET_DISTANCE_CM = 25.0

STEP_DURATION_S = 0.5        # how long to apply a step command before stopping to look again
STEP_SPEED = 20.0            # TODO tune: hand-tested speed for a short step toward the target
LOOK_PAUSE_S = 0.2           # how long to wait after stopping before looking

# Deadband prevents oscillation around target distance.
TOO_CLOSE_DISTANCE_CM = 23.0
TOO_FAR_DISTANCE_CM = 27.0


# When distance is correct, rotate instead of driving forward.
ALIGN_TURN_SPEED = 15.0

# Minimum camera error before considering aligned.
ALIGNMENT_THRESHOLD = 0.05


def compute_reactive_command(
    error: TargetError,
    ultrasonic_distance: float | None,
    kin: DifferentialKinematics
) -> Optional[WheelCommand]:

    if not error.found:
        return None


    # Emergency visual backup
    if error.too_close:
        print("[STOP] TOO CLOSE")
        return kin.backward(speed=BACKUP_SPEED)


    if error.reached:
        return None


    # ------------------------------------------------
    # Distance regulation using ultrasonic
    # ------------------------------------------------

    if ultrasonic_distance is not None:


        # Too close
        if ultrasonic_distance < TOO_CLOSE_DISTANCE_CM:

            print("[CTRL] TOO CLOSE - BACKING")

            if error.lateral_error > 0:
                return kin.arc_backward_left(
                    angle_deg=30,
                    speed=BACKUP_SPEED
                )

            else:
                return kin.arc_backward_right(
                    angle_deg=30,
                    speed=BACKUP_SPEED
                )


        # Too far
        elif ultrasonic_distance > TOO_FAR_DISTANCE_CM:

            print("[CTRL] APPROACH")

            steer_angle = min(
                MAX_STEER_ANGLE_DEG,
                abs(error.lateral_error)
                * 2
                * MAX_STEER_ANGLE_DEG
            )


            if error.lateral_error > 0:

                return kin.arc_forward_left(
                    angle_deg=steer_angle,
                    speed=APPROACH_SPEED
                )

            else:

                return kin.arc_forward_right(
                    angle_deg=steer_angle,
                    speed=APPROACH_SPEED
                )


        # Correct distance
        else:

            print("[CTRL] ALIGNMENT MODE")

            return compute_alignment_command(
                error,
                kin
            )


    # ------------------------------------------------
    # Fallback if ultrasonic unavailable
    # ------------------------------------------------

    print("[CTRL] ULTRASONIC UNKNOWN")

    if error.lateral_error > 0:

        return kin.arc_forward_left(
            angle_deg=30,
            speed=FALLBACK_SPEED
        )

    else:

        return kin.arc_forward_right(
            angle_deg=30,
            speed=FALLBACK_SPEED
        )


def speed_tier(error: TargetError) -> str:
    """Which branch of compute_reactive_command's logic this error would hit,
    as a short label -- for logging/display only (e.g. the live loop's
    console/overlay monitor). Mirrors that function's branching exactly, so
    the readout can't silently drift out of sync with the actual thresholds
    as they get re-tuned; keep the two in sync if the branches ever change.

    One of: "none" (not found), "reached", "backup" (too_close), "fallback"
    (distance_cm unknown), "cruise", "approach", "step" (the
    is_final_approach() zone).
    """
    if not error.found:
        return "none"
    if error.too_close:
        return "backup"
    if error.reached:
        return "reached"
    if error.distance_cm is None:
        return "fallback"
    if error.distance_cm >= CRUISE_DISTANCE_CM:
        return "cruise"
    return "step"


def compute_alignment_command(
    error: TargetError,
    kin: DifferentialKinematics
) -> WheelCommand:
    """
    At correct distance, only fix orientation.

    No forward/backward motion.
    Camera controls left/right rotation.
    """

    if abs(error.lateral_error) <= ALIGNMENT_THRESHOLD:
        # Already centered
        return kin.forward(speed=0)

    if error.lateral_error > 0:
        # Object appears left
        return kin.turn_left(speed=ALIGN_TURN_SPEED)

    else:
        # Object appears right
        return kin.turn_right(speed=ALIGN_TURN_SPEED)


def is_final_approach(error: TargetError) -> bool:
    """True once close enough (inside FAR_DISTANCE_CM) that the caller should
    switch from driving compute_reactive_command's output continuously to
    short step-then-look bursts instead: apply the command for
    STEP_DURATION_S, stop, wait LOOK_PAUSE_S for a fresh unblurred frame,
    then re-evaluate -- rather than driving on a frame that's already stale
    by the time it reaches the wheels, which this close is enough to
    overshoot the grab point or drift off-center. The command itself doesn't
    change (same speed-tiered/steered WheelCommand either way); only how
    often the caller applies it does, so the timing lives in the live loop,
    not here.

    False once already reached (nothing left to approach), too_close
    (backing off takes priority over fine positioning), or still far enough
    out that continuous cruising is fine and faster.
    """
    return (error.found and not error.reached and not error.too_close
            and error.distance_cm is not None)
