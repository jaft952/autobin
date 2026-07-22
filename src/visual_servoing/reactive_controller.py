"""
src/visual_servoing/reactive_controller.py

Target error -> WheelCommand, via DifferentialKinematics. Pure function: no
hardware access here (Rule: hardware access only through src/hardware).

Also exposes is_final_approach(), a pure predicate the live loop uses to
decide WHEN to switch from driving compute_reactive_command's output
continuously (cruising) to applying it in short step-then-look bursts
instead (inside FAR_DISTANCE_CM) -- see that function's docstring. The
timing itself (STEP_DURATION_S / LOOK_PAUSE_S) is the caller's job, same as
the rest of hardware/timing.
"""
from __future__ import annotations

from typing import Optional

from src.visual_servoing.distance_error import TargetError, STOP_DISTANCE_CM
from src.motion.differential_kinematics import DifferentialKinematics, WheelCommand

MAX_SPEED = 22.0             # TODO tune: matches MotionCalibration's default arc_speed
FALLBACK_SPEED = 22.0        # TODO tune: used only when distance can't be estimated at all
BACKUP_SPEED = 22.0          # TODO tune: gentle reverse to recover from an overshoot
MAX_STEER_ANGLE_DEG = 50.0   # keep below 90 so it never fully spins in place
FAR_DISTANCE_CM = 30.0       # distance at/beyond which speed is MAX_SPEED

# Below FAR_DISTANCE_CM, continuous driving risks overshoot: by the time a
# command reaches the wheels the frame it was computed from is already
# stale, and this close, a stale frame's worth of travel is enough to blow
# past the grab point or drift off-center. is_final_approach() flags this
# zone; the caller (the live loop) is responsible for actually stepping
# instead of driving continuously -- see STEP_DURATION_S / LOOK_PAUSE_S
# below, used there, not here (this module stays hardware/time-free).
STEP_DURATION_S = 0.25       # TODO tune: length of one forward/steer nudge
LOOK_PAUSE_S = 0.6           # TODO tune: stopped time for a fresh, unblurred look


def compute_reactive_command(error: TargetError,
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
        dynamic_speed = MAX_SPEED * frac # -> 0 right at STOP_DISTANCE_CM

    if error.lateral_error > 0:
        return kin.arc_forward_left(angle_deg=steer_angle, speed=dynamic_speed)
    else:
        return kin.arc_forward_right(angle_deg=steer_angle, speed=dynamic_speed)


def is_final_approach(error: TargetError) -> bool:
    """True once close enough (inside FAR_DISTANCE_CM) that the caller should
    switch from driving compute_reactive_command's output continuously to
    short step-then-look bursts instead: apply the command for
    STEP_DURATION_S, stop, wait LOOK_PAUSE_S for a fresh unblurred frame,
    then re-evaluate -- rather than driving on a frame that's already stale
    by the time it reaches the wheels, which this close is enough to
    overshoot the grab point or drift off-center. The command itself doesn't
    change (same speed-ramped/steered WheelCommand either way); only how
    often the caller applies it does, so the timing lives in the live loop,
    not here.

    False once already reached (nothing left to approach), too_close
    (backing off takes priority over fine positioning), or still far enough
    out that continuous cruising is fine and faster.
    """
    return (error.found and not error.reached and not error.too_close
            and error.distance_cm is not None
            and error.distance_cm <= FAR_DISTANCE_CM)
