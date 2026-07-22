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

MAX_STEER_ANGLE_DEG = 80.0   # keep below 90 so it never fully spins in place
FAR_DISTANCE_CM = 30.0       # distance at/below which is_final_approach() triggers step-and-look

# Below FAR_DISTANCE_CM, continuous driving risks overshoot: by the time a
# command reaches the wheels the frame it was computed from is already
# stale, and this close, a stale frame's worth of travel is enough to blow
# past the grab point or drift off-center. is_final_approach() flags this
# zone; the caller (the live loop) is responsible for actually stepping
# instead of driving continuously -- see STEP_DURATION_S / LOOK_PAUSE_S
# below, used there, not here (this module stays hardware/time-free).
STEP_SPEED = 25.0            # TODO tune: hand-tested fixed nudge speed for step-and-look
STEP_DURATION_S = 0.25       # TODO tune: length of one forward/steer nudge
LOOK_PAUSE_S = 0.6           # TODO tune: stopped time for a fresh, unblurred look


def compute_reactive_command(error: TargetError,
                           kin: DifferentialKinematics) -> Optional[WheelCommand]:
    """
    Speed is a lookup against hardcoded distance breakpoints, not a
    continuous formula (see the constants block above for why):
        distance_cm >= CRUISE_DISTANCE_CM  (~90cm) -> MAX_SPEED
        FAR_DISTANCE_CM < distance_cm < CRUISE_DISTANCE_CM (~60cm) -> APPROACH_SPEED
        distance_cm <= FAR_DISTANCE_CM     (~30cm) -> STEP_SPEED

    That last tier never reaches zero -- unlike the old zero-at-STOP_DISTANCE_CM
    ramp this replaced, steering keeps working the whole way to the grab
    point since dynamic_speed (which both forward speed AND steer angle scale
    off, via DifferentialKinematics.arc_forward_*) never collapses to
    nothing. The caller applies this tier as a short step-and-look nudge
    rather than driving it continuously -- see is_final_approach().

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
        dynamic_speed = FALLBACK_SPEED
    elif error.distance_cm >= CRUISE_DISTANCE_CM:
        dynamic_speed = MAX_SPEED
    elif error.distance_cm > FAR_DISTANCE_CM:
        dynamic_speed = APPROACH_SPEED
    else:
        dynamic_speed = STEP_SPEED   # is_final_approach() zone

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
    if error.distance_cm > FAR_DISTANCE_CM:
        return "approach"
    return "step"


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
            and error.distance_cm is not None
            and error.distance_cm <= FAR_DISTANCE_CM)
