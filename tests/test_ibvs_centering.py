"""
tests/test_ibvs_centering.py

Pure-logic tests for distance_error.py + reactive_controller.py. NO hardware or
camera needed — builds synthetic DetectionResult/BoundingBox fixtures by
hand, same spirit as tests/test_scan_logic.py:

    python tests/test_ibvs_centering.py      # plain runner with PASS/FAIL summary
    pytest tests/test_ibvs_centering.py      # also works

For a live run on the robot (real camera, optionally real motors/arm) use:

    python tests/test_ibvs_centering.py --live            # camera + overlay only, nothing moves
    python tests/test_ibvs_centering.py --live --drive    # also sends commands to the wheels
    python tests/test_ibvs_centering.py --live --drive --arm   # + grabs when "reached"

Live flags (only read with --live):
  --drive              send commands to the wheels (default: look only)
  --arm                grab when reached (needs a calibrated arc_grasp)

Without --drive the wheels never move — distance_error and reactive_controller
still run every frame and their output shows on the overlay/console, so the
planned steer/speed can be checked before it touches the motors. Keys while the
window is focused: m = toggle drive, g = toggle arm, q = quit.

The live demo is NOT executed by the test suite or by pytest — see
run_live_demo() at the bottom.
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.perception.detector import BoundingBox, DetectionResult
from src.motion.calibration import MotionCalibration
from src.motion.differential_kinematics import DifferentialKinematics
from src.arm.arc_grasp import (
    pose_and_angle, row_ny_at, NX_TOL_DEFAULT, BAND_TOO_CLOSE, BAND_TOO_FAR,
    BAND_NX_OUTSIDE, BAND_NOT_CALIBRATED)
from src.visual_servoing.distance_error import compute_target_error, TargetError
import src.hardware.actuators.pwm_driver as pwm_driver
import src.visual_servoing.reactive_controller as rc
from src.visual_servoing.reactive_controller import (
    compute_reactive_command, is_final_approach, speed_tier,
    MAX_SPEED, APPROACH_SPEED, STEP_SPEED, BACKUP_SPEED, STEP_DURATION_S, LOOK_PAUSE_S,
    RETREAT_PULSE_S, ALIGN_TURN_SPEED, ALIGN_PULSE_S,
)
from src.visual_servoing.ultrasonic_safety import (
    UltrasonicSafety, UltrasonicWatchdog, EMERGENCY_STOP_CM, GRAB_CONFIRM_CM)


# ── Fixture builder ──────────────────────────────────────────────────────

def _make_detection(norm_x: float, bbox_height_px: float,
                     frame_width: int = 640, frame_height: int = 480) -> DetectionResult:
    """A single detection whose ground-contact point sits at norm_x (0..1
    across the frame) with the given pixel bbox height (drives the monocular
    distance estimate)."""
    center_x = norm_x * frame_width
    half_w = 30
    x1 = int(center_x - half_w)
    x2 = int(center_x + half_w)
    y1 = 100
    y2 = int(y1 + bbox_height_px)
    box = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=0.9)
    return DetectionResult(detections=[box], frame_width=frame_width, frame_height=frame_height)


def _kin() -> DifferentialKinematics:
    return DifferentialKinematics(MotionCalibration())


# ── distance_error.compute_target_error ──────────────────────────────────

def test_no_detection_not_found():
    result = DetectionResult()   # no detections at all
    error = compute_target_error(result)
    assert not error.found
    assert error.distance_cm is None
    assert not error.reached
    print("PASS no detection -> not found")


def test_far_left_of_center():
    """Can left of center, far away -> negative lateral_error, large distance."""
    result = _make_detection(norm_x=0.3, bbox_height_px=40)
    error = compute_target_error(result)
    assert error.found
    assert error.lateral_error < 0, error
    assert error.distance_cm is not None and error.distance_cm > 30, error
    assert not error.reached
    print("PASS far + left of center")


def test_close_right_of_center():
    """Can right of center, close but not yet at the stop distance."""
    result = _make_detection(norm_x=0.7, bbox_height_px=200)
    error = compute_target_error(result)
    assert error.found
    assert error.lateral_error > 0, error
    assert error.distance_cm is not None and 15 < error.distance_cm < 80, error
    assert not error.reached
    print("PASS close + right of center")


def test_reached_when_centered_and_close():
    """Centered and within STOP_DISTANCE_CM -> reached, ready for arm handoff.
    bbox_height_px chosen to stay clear of CLOSE_BBOX_FRACTION (300/480=0.625
    < 0.75) so this exercises "reached" in isolation from "too_close"."""
    result = _make_detection(norm_x=0.5, bbox_height_px=300)
    error = compute_target_error(result)
    assert error.found
    assert abs(error.lateral_error) < 1e-9, error
    assert error.distance_cm is not None and error.distance_cm <= 15, error
    assert error.reached
    assert not error.too_close, error
    print("PASS reached: centered + close")


def test_too_close_backs_away_even_when_off_center():
    """Bbox fills most of the frame (360/480=0.75, at CLOSE_BBOX_FRACTION) ->
    too_close True regardless of the (possibly uncalibrated) distance_cm
    math, and regardless of centering. This is the failsafe: a wrong
    CALIBRATION_CONSTANT_PX_CM must not be able to wedge the robot against
    the can — it should back off instead."""
    result = _make_detection(norm_x=0.3, bbox_height_px=360)
    error = compute_target_error(result)
    assert error.found
    assert error.too_close, error
    assert not error.reached, error   # off-center, so not "grasp-ready"
    print("PASS too_close (checked further in the drive-command tests below)")


# ── reactive_controller.compute_reactive_command ─────────────────────────────────

def test_reactive_command_none_when_not_found():
    error = compute_target_error(DetectionResult())
    assert compute_reactive_command(error, _kin()) is None
    print("PASS drive command is None when nothing is found")


def test_reactive_command_none_when_reached():
    error = compute_target_error(_make_detection(norm_x=0.5, bbox_height_px=300))
    assert error.reached
    assert not error.too_close, error
    assert compute_reactive_command(error, _kin()) is None
    print("PASS drive command is None when reached (arm handoff)")


def test_reactive_command_backs_away_when_too_close():
    """too_close -> an active BACKWARD command (both wheels negative), not
    just a stop — recovers from an overshoot instead of sitting wedged
    against the can, even when off-center."""
    error = compute_target_error(_make_detection(norm_x=0.3, bbox_height_px=360))
    assert error.too_close
    cmd = compute_reactive_command(error, _kin())
    assert cmd is not None
    assert cmd.left_speed < 0 and cmd.right_speed < 0, cmd
    assert abs(cmd.left_speed) <= BACKUP_SPEED + 1e-6, cmd
    print("PASS backs away (both wheels reverse) when too_close")


def test_reactive_command_when_can_is_left_of_center():
    """Far + left of center -> steers toward arc_forward_right (right wheel
    slower than left) — hardware-verified mapping, see the sign-convention
    note in reactive_controller.py. Far away (>= CRUISE_DISTANCE_CM) ->
    speed at MAX_SPEED, the top hardcoded tier."""
    error = compute_target_error(_make_detection(norm_x=0.3, bbox_height_px=40))
    cmd = compute_reactive_command(error, _kin())
    assert cmd is not None
    assert cmd.right_speed < cmd.left_speed, cmd
    assert cmd.left_speed >= 0.9 * MAX_SPEED, cmd   # brisk, not crawling
    print("PASS steers toward the can + full speed when far + left of center")


def test_reactive_command_when_can_is_right_of_center():
    """Inside FAR_DISTANCE_CM (final approach / step-and-look) + right of
    center -> steers toward arc_forward_left (left wheel slower than right)
    — hardware-verified mapping, see reactive_controller.py — at the fixed
    STEP_SPEED nudge, not a ramped-to-zero value (no more zero-speed
    stalling: STEP_SPEED keeps steering alive all the way to the grab
    point). frame_height oversized per the is_final_approach fixtures above
    to stay clear of CLOSE_BBOX_FRACTION at this bbox size."""
    error = compute_target_error(_make_detection(norm_x=0.7, bbox_height_px=1064, frame_height=5000))
    assert is_final_approach(error), error
    cmd = compute_reactive_command(error, _kin())
    assert cmd is not None
    assert cmd.left_speed < cmd.right_speed, cmd
    assert abs(cmd.right_speed - STEP_SPEED) < 0.2 * STEP_SPEED, cmd   # right (outer) wheel ~= STEP_SPEED * trim
    print("PASS steers toward the can + step-and-look nudge speed when in final approach")


def test_reactive_command_speed_tiers_are_hardcoded():
    """Same lateral offset, three distances spanning the three breakpoints
    -> each picks a distinct hardcoded tier (per speed_tier()'s label) with
    the outer wheel's speed matching that tier's constant, confirming a
    lookup against CRUISE_DISTANCE_CM / FAR_DISTANCE_CM rather than a
    continuous formula. Tolerant of the tiers currently being hand-tuned to
    the same numeric value -- the point is which constant got looked up,
    not that the values must differ."""
    cruise_error = compute_target_error(_make_detection(norm_x=0.7, bbox_height_px=40))        # ~142cm
    approach_error = compute_target_error(_make_detection(norm_x=0.7, bbox_height_px=164))      # ~70cm
    step_error = compute_target_error(_make_detection(norm_x=0.7, bbox_height_px=1064, frame_height=5000))  # ~27cm

    assert speed_tier(cruise_error) == "cruise", cruise_error
    assert speed_tier(approach_error) == "approach", approach_error
    assert speed_tier(step_error) == "step", step_error

    cruise = compute_reactive_command(cruise_error, _kin())
    approach = compute_reactive_command(approach_error, _kin())
    step = compute_reactive_command(step_error, _kin())
    assert cruise is not None and approach is not None and step is not None
    assert abs(cruise.right_speed - MAX_SPEED) < 0.2 * MAX_SPEED, cruise
    assert abs(approach.right_speed - APPROACH_SPEED) < 0.2 * APPROACH_SPEED, approach
    assert abs(step.right_speed - STEP_SPEED) < 0.2 * STEP_SPEED, step
    print("PASS speed is a hardcoded tier lookup keyed by distance breakpoints, not a continuous ramp")


def test_speed_tier_labels_cover_every_branch():
    """speed_tier() returns the exact branch label compute_reactive_command
    would take for the same error -- the log/overlay's single source of
    truth for which tier is active. distance_cm=None ('fallback') is
    awkward to hit through compute_target_error, so that one TargetError is
    built directly instead of through a synthetic detection."""
    assert speed_tier(compute_target_error(DetectionResult())) == "none"
    assert speed_tier(compute_target_error(
        _make_detection(norm_x=0.3, bbox_height_px=360))) == "backup"     # too_close
    assert speed_tier(compute_target_error(
        _make_detection(norm_x=0.5, bbox_height_px=1662, frame_height=5000))) == "reached"
    assert speed_tier(compute_target_error(
        _make_detection(norm_x=0.3, bbox_height_px=40))) == "cruise"      # ~142cm
    assert speed_tier(compute_target_error(
        _make_detection(norm_x=0.3, bbox_height_px=164))) == "approach"   # ~70cm
    assert speed_tier(compute_target_error(
        _make_detection(norm_x=0.5, bbox_height_px=1064, frame_height=5000))) == "step"   # ~27cm
    fallback_error = TargetError(found=True, lateral_error=0.0, distance_cm=None,
                                  reached=False, too_close=False)
    assert speed_tier(fallback_error) == "fallback"
    print("PASS speed_tier labels match every compute_reactive_command branch")


# ── reactive_controller.is_final_approach ────────────────────────────────
#
# Fixtures below use an oversized frame_height (5000px) purely to keep
# bbox_height_px/frame_height under CLOSE_BBOX_FRACTION while still hitting
# the target distance_cm band under today's (uncalibrated, see
# distance_error.py) CALIBRATION_CONSTANT_PX_CM placeholder -- these pixel
# numbers aren't meant to look like a real camera frame.

def test_final_approach_when_close_but_not_reached():
    """Inside FAR_DISTANCE_CM but outside STOP_DISTANCE_CM -> switch to
    step-then-look (centered so the distance zone is isolated from
    centering as the reason)."""
    result = _make_detection(norm_x=0.5, bbox_height_px=1064, frame_height=5000)
    error = compute_target_error(result)
    assert error.found and not error.reached and not error.too_close
    assert is_final_approach(error), error
    print("PASS final approach: close but not yet at the stop distance")


def test_cruise_when_far():
    """Far away -> still cruising, not yet stepping."""
    result = _make_detection(norm_x=0.3, bbox_height_px=40)
    error = compute_target_error(result)
    assert not is_final_approach(error), error
    print("PASS cruise (not final approach) when far")


def test_not_final_approach_once_reached():
    """Already reached -> nothing left to step toward."""
    result = _make_detection(norm_x=0.5, bbox_height_px=1662, frame_height=5000)
    error = compute_target_error(result)
    assert error.reached
    assert not is_final_approach(error), error
    print("PASS not final-approach once already reached")


def test_not_final_approach_when_not_found():
    error = compute_target_error(DetectionResult())
    assert not is_final_approach(error)
    print("PASS not final-approach when nothing is found")


def test_not_final_approach_when_too_close():
    """too_close takes priority over step-and-look -- backing away wins."""
    result = _make_detection(norm_x=0.3, bbox_height_px=360)
    error = compute_target_error(result)
    assert error.too_close
    assert not is_final_approach(error), error
    print("PASS not final-approach when too_close (backing away takes priority)")


# ── Plain runner (no pytest needed) ───────────────────────────────────────

ALL_TESTS = [
    test_no_detection_not_found,
    test_far_left_of_center,
    test_close_right_of_center,
    test_reached_when_centered_and_close,
    test_too_close_backs_away_even_when_off_center,
    test_reactive_command_none_when_not_found,
    test_reactive_command_none_when_reached,
    test_reactive_command_backs_away_when_too_close,
    test_reactive_command_when_can_is_left_of_center,
    test_reactive_command_when_can_is_right_of_center,
    test_reactive_command_speed_tiers_are_hardcoded,
    test_speed_tier_labels_cover_every_branch,
    test_final_approach_when_close_but_not_reached,
    test_cruise_when_far,
    test_not_final_approach_once_reached,
    test_not_final_approach_when_not_found,
    test_not_final_approach_when_too_close,
]


def _run_all_tests():
    failed = []
    for fn in ALL_TESTS:
        try:
            fn()
        except AssertionError as exc:
            failed.append(fn.__name__)
            print(f"FAIL {fn.__name__}: {exc}")
    print()
    if failed:
        print(f"{len(failed)}/{len(ALL_TESTS)} FAILED: {', '.join(failed)}")
        sys.exit(1)
    print(f"all {len(ALL_TESTS)} tests passed")


# ── Live hardware demo (NOT run by the test suite / pytest / an agent) ───

_GRAB_COOLDOWN_S = 4.0   # after a grab attempt (success or refusal), don't re-solve every frame

# How long a "solver says grabbable" answer keeps the wheels stopped after
# the solver stops saying it -- rides out single-frame detection jitter so a
# blink doesn't restart the approach.
GRABBABLE_LATCH_S = 2.0

# Caller-side band value: nothing detected this frame. Distinct from
# BAND_NOT_CALIBRATED (no solver at all) so the overlay can't blame a missing
# grid for what is really an empty frame.
BAND_NO_TARGET = "no_target"


# Display-only: color per speed_tier() label, used by both the overlay and
# (as a quick reference) anyone reading this file. Keep the KEYS in sync
# with reactive_controller.speed_tier()'s possible return values -- an
# unrecognized tier just falls back to TIER_COLOR's default below rather
# than raising, since this is a monitor, not a safety path.
TIER_COLOR = {
    "cruise": (0, 255, 0),      # green   -- brisk, MAX_SPEED
    "approach": (0, 220, 170),  # teal    -- APPROACH_SPEED
    "step": (0, 200, 255),      # orange  -- STEP_SPEED, step-and-look nudge
    "backup": (0, 140, 255),    # red-orange -- BACKUP_SPEED, too_close
    "reached": (0, 255, 0),     # green   -- arm handoff point
    "fallback": (0, 165, 255),  # amber   -- FALLBACK_SPEED, distance unknown
    "none": (0, 0, 255),        # red     -- nothing found
}


def _draw_status_overlay(frame, error, cmd, driving: bool, armed: bool,
                         bbox_area_px=None, controller_name: str = "reactive",
                         tier: str = "none", recovered_nudge: bool = False,
                         ultra=None, solved=None, grabbable: bool = False,
                         band: str = BAND_NOT_CALIBRATED, applied=None) -> None:
    """Burn the distance_error / drive-command readout onto the frame so the
    planned IK output (steer + dynamic speed) is visible without reading the
    console, and so a bad estimate is obvious immediately.

    tier: reactive_controller.speed_tier(error)'s label for this frame --
    the single source of truth for which hardcoded speed constant is
    active, so this overlay can't drift out of sync with
    compute_reactive_command's actual thresholds as they get re-tuned.

    recovered_nudge: True for the one frame where the live loop fired the
    "can was pushed closer while reached" backward nudge (see
    run_live_demo) -- cmd is still None that frame (compute_reactive_command
    doesn't know about the nudge), so without this flag the overlay would
    misleadingly show a plain STOPPED.

    bbox_area_px is shown raw (not just the derived distance_cm) because
    CALIBRATION_CONSTANT_PX_CM starts as an unmeasured placeholder — this is
    the number you read off the screen with the can at a known distance to
    calibrate it (see distance_error.py).

    ultra: the watchdog's latest UltrasonicState, or None if unavailable —
    shown as its own line so a dead/miswired sensor (reading stays "--") is
    distinguishable from a live one that simply isn't tripping yet.

    solved: arc_grasp's live solution this frame (None = not in the band).
    grabbable: the latched version that actually stops the wheels — shown
    separately so a latch riding out a detection blink is visible as such."""
    import cv2
    fh, fw = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    banner_h = 162
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, fh - banner_h), (fw, fh), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, dst=frame)

    if not error.found:
        target_line, target_color = "TARGET: none", (0, 0, 255)
    else:
        dist_txt = f"{error.distance_cm:.1f}" if error.distance_cm is not None else "?"
        bbox_txt = f"{bbox_area_px}px" if bbox_area_px is not None else "?"
        target_line = (f"TARGET: lateral={error.lateral_error:+.2f} "
                        f"dist_cm={dist_txt} bbox_a={bbox_txt} "
                        f"reached={error.reached} too_close={error.too_close}")
        target_color = (0, 255, 0) if error.reached else \
            ((0, 140, 255) if error.too_close else (0, 255, 255))
    cv2.putText(frame, target_line, (16, fh - banner_h + 26), font, 0.55, target_color, 2)

    if cmd is None:
        if recovered_nudge:
            reason = "reached, backed off (can moved closer)"
        elif tier == "reached":
            reason = "reached"
        elif tier == "none":
            reason = "no target"
        else:
            reason = "stopped"
        drive_line = f"DRIVE: STOPPED ({reason})"
        drive_color = (0, 140, 255) if recovered_nudge else (170, 170, 170)
    elif cmd.left_speed < 0 and cmd.right_speed < 0:
        drive_line = (f"DRIVE [BACKUP]: BACKING UP (too_close) "
                      f"L={cmd.left_speed:+.1f} R={cmd.right_speed:+.1f}")
        drive_color = TIER_COLOR["backup"]
    else:
        # Derived from lateral_error's sign (which way the CAN is, and so
        # which way we're correcting), not from comparing wheel speeds —
        # that comparison depends on which kin method reactive_controller.py
        # happens to call for a given sign, which is hardware-specific and
        # has already flipped once (see the note in reactive_controller.py).
        if error.lateral_error > 0.02:
            steer = "RIGHT"
        elif error.lateral_error < -0.02:
            steer = "LEFT"
        else:
            steer = "STRAIGHT"
        drive_line = (f"DRIVE [{tier.upper()}]: L={cmd.left_speed:+.1f} R={cmd.right_speed:+.1f} "
                      f"(steer={steer})")
        drive_color = TIER_COLOR.get(tier, (0, 255, 0))
    if applied is not None:
        # Post-trim/floor/clamp duty actually sent to the motors -- the number
        # that has to clear stiction, which the command above may not equal.
        drive_line += f"  ->PWM L={applied[0]:+.1f} R={applied[1]:+.1f}"
    cv2.putText(frame, drive_line, (16, fh - banner_h + 54), font, 0.6, drive_color, 2)

    mode_line = (f"[m] drive={'ON' if driving else 'OFF'}  [g] arm={'ARMED' if armed else 'OFF'}  "
                 f"[l] lines  [ [ ] ] floor={pwm_driver.MIN_MOVE_DUTY:.0f}  "
                 f"[ - = ] step={rc.STEP_SPEED:.0f}  [q] quit")
    cv2.putText(frame, mode_line, (16, fh - banner_h + 80), font, 0.5, (200, 200, 200), 1)

    if ultra is None or ultra.distance_cm is None:
        ultra_line = "ULTRA: no reading (sensor dead/out of range?)"
        ultra_color = (0, 0, 255)
    else:
        ultra_line = (f"ULTRA: {ultra.distance_cm:.1f}cm  "
                      f"estop={ultra.emergency_stop} (<={EMERGENCY_STOP_CM:.0f})  "
                      f"grab_ok={ultra.grab_confirmed} (<={GRAB_CONFIRM_CM:.0f})")
        ultra_color = (0, 140, 255) if ultra.emergency_stop else \
            ((0, 255, 0) if ultra.grab_confirmed else (200, 200, 200))
    cv2.putText(frame, ultra_line, (16, fh - banner_h + 106), font, 0.55, ultra_color, 2)

    if solved is not None:
        grab_line = ("GRABBABLE: YES  arm=["
                     + " ".join(f"{v:.0f}" for v in solved) + "]")
        grab_color = (0, 255, 0)
    elif grabbable:
        grab_line = "GRABBABLE: latched (solver blinked, holding stop)"
        grab_color = (0, 255, 255)
    else:
        why = {
            BAND_TOO_FAR: "too far -- keep approaching",
            BAND_TOO_CLOSE: "OVERSHOT past nearest arc -- backing off",
            BAND_NX_OUTSIDE: "nx off sampled span -- turning in place",
            BAND_NOT_CALIBRATED: "NO SOLVER -- arc grid missing/uncalibrated",
            BAND_NO_TARGET: "no tin detected",
        }.get(band, band)
        grab_line = f"GRABBABLE: no ({why})"
        if band == BAND_NOT_CALIBRATED:
            grab_color = (0, 0, 255)    # red: band logic is off entirely
        elif band == BAND_TOO_CLOSE:
            grab_color = (0, 140, 255)
        else:
            grab_color = (170, 170, 170)
    cv2.putText(frame, grab_line, (16, fh - banner_h + 132), font, 0.55, grab_color, 2)

    dcol = (0, 255, 0) if driving else (0, 0, 255)
    cv2.putText(frame, "DRIVE ON" if driving else "DRIVE OFF", (fw - 190, 30), font, 0.65, (0, 0, 0), 3)
    cv2.putText(frame, "DRIVE ON" if driving else "DRIVE OFF", (fw - 190, 30), font, 0.65, dcol, 1)
    if armed:
        cv2.putText(frame, "ARM ARMED", (fw - 190, 58), font, 0.65, (0, 0, 0), 3)
        cv2.putText(frame, "ARM ARMED", (fw - 190, 58), font, 0.65, (0, 255, 0), 1)


_BAND_MASK_CACHE: dict = {}


def _band_mask(solver, pose: str, fw: int, fh: int, step: int = 8):
    """Binary mask of every image point where solve() succeeds.

    Built by probing the solver on a coarse grid, so the grabbable region is
    something you LOOK at instead of infer from a text label -- a row whose
    nx span is one sample shows up immediately as a sliver. Cached per
    (pose, size); the grid only changes when the yaml is re-read, which the
    live loop never does."""
    import numpy as np
    key = (pose, fw, fh, step)
    hit = _BAND_MASK_CACHE.get(key)
    if hit is not None:
        return hit
    mask = np.zeros((fh, fw), dtype=np.uint8)
    for py in range(0, fh, step):
        ny = (py + step * 0.5) / fh
        for px in range(0, fw, step):
            nx = (px + step * 0.5) / fw
            if solver.solve(nx, ny, pose=pose) is not None:
                mask[py:py + step, px:px + step] = 255
    _BAND_MASK_CACHE[key] = mask
    return mask


def _draw_arc_lines(frame, solver, pose: str, point=None) -> None:
    """Draw the arc_grasp grid: solvable region tinted, each row's curve, and
    the sampled nx span that curve is actually usable over.

    A row is drawn dim where solve() would refuse it for nx (outside the
    sampled span +/- nx_tol) and bright where it is usable -- the gap between
    those two is what silently returns BAND_NX_OUTSIDE.
    """
    import cv2
    import numpy as np
    rows = solver.rows_for(pose)
    if not rows:
        return
    fh, fw = frame.shape[:2]

    mask = _band_mask(solver, pose, fw, fh)
    tint = np.full_like(frame, (0, 170, 0))
    cv2.addWeighted(cv2.bitwise_and(tint, tint, mask=mask), 0.30, frame, 1.0, 0,
                    dst=frame)

    for i, r in enumerate(rows):
        default_ny = float(r["ny"])
        samples = sorted((float(s["nx"]), float(s.get("ny", default_ny)))
                         for s in r["samples"])
        tol = float(r.get("nx_tol", NX_TOL_DEFAULT))
        lo_nx, hi_nx = samples[0][0] - tol, samples[-1][0] + tol

        curve = [(px, int(row_ny_at(r, px / fw) * fh))
                 for px in range(0, fw, 6)]
        cv2.polylines(frame, [np.array(curve, np.int32)], False, (90, 90, 90), 1)
        usable = [p for p in curve if lo_nx <= p[0] / fw <= hi_nx]
        if len(usable) > 1:
            cv2.polylines(frame, [np.array(usable, np.int32)], False,
                          (0, 220, 220), 2)
        elif usable:                       # single-sample row: no span at all
            x0, y0 = usable[0]
            cv2.line(frame, (x0 - 6, y0), (x0 + 6, y0), (0, 140, 255), 2)

        for nx, ny in samples:
            cv2.drawMarker(frame, (int(nx * fw), int(ny * fh)), (0, 255, 255),
                           cv2.MARKER_DIAMOND, 10, 1)
        lx, ly = int(samples[0][0] * fw), int(samples[0][1] * fh)
        cv2.putText(frame, f"r{i} ny={default_ny:.3f} n={len(samples)}",
                    (max(4, lx - 30), max(12, ly - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 220), 1)

    if point is not None:
        cv2.drawMarker(frame, (int(point[0] * fw), int(point[1] * fh)),
                       (255, 0, 255), cv2.MARKER_TILTED_CROSS, 18, 2)


def _solve_for_detection(result: DetectionResult, solver):
    """(solved_CH1_5_or_None, tin_pose, point, band) for the current detection.

    The single grabbability check: the live loop calls this every frame to
    decide when to STOP driving, and the same answer runs the grab. Driving
    used to stop on error.reached instead, which is a bbox-area threshold
    against an uncalibrated constant and knows nothing about arc_grasp.yaml
    -- so the robot drove past perfectly grabbable rows until the ultrasonic
    tripped. band says WHY a None is None, so overshooting the nearest arc
    can back off instead of closing in further."""
    if solver is None:
        return None, None, None, BAND_NOT_CALIBRATED
    if result.best is None:
        return None, None, None, BAND_NO_TARGET
    pose, angle = pose_and_angle(result.best)
    point = result.normalized_center() if pose == "lying" else result.normalized_base_center()
    if point is None:
        return None, pose, None, BAND_NO_TARGET
    solved, band = solver.solve_with_band(point[0], point[1],
                                          pose=pose, angle_deg=angle)
    return solved, pose, point, band


def _attempt_grab(solved, pose, point, arm, actuator, state: dict) -> None:
    """Run the SAME collect (grab -> dump) -> home sequence the autonomous
    stack runs, so this is byte-for-byte the grab you tune in the calibration
    tool. solved/pose/point come from _solve_for_detection (already checked
    non-None by the caller). Cooldown-gated so a tin sitting in the band
    doesn't re-trigger every single frame."""
    import time as _time
    if _time.monotonic() < state["cooldown_until"]:
        return
    state["cooldown_until"] = _time.monotonic() + _GRAB_COOLDOWN_S

    print(f"[arm] GRAB ({pose}) @ nx={point[0]:.2f} ny={point[1]:.2f}")
    if not state["homed"]:
        arm.goto("home")   # tracked pose may be from a stale prior session
        state["homed"] = True
    # Hold the base ourselves: the arm shakes the chassis, and coasting
    # wheels would drift the robot off the solved spot mid-grab.
    actuator.brake()
    try:
        arm.collect(solved, tin_pose=pose)   # grab + drop in the bin
        arm.goto("home")
        arm.release()
    finally:
        actuator.stop()
    print("[arm] grab sequence done.")


def run_live_demo():
    """Live camera view (segmentation + bbox + base-contact point, same as
    the perception module's own annotator) with the distance_error /
    reactive_controller readout burned onto the frame, so the planned IK output
    can be checked visually before it ever touches the motors.

    Run by hand on the Pi:
        python tests/test_ibvs_centering.py --live              # look only, nothing moves
        python tests/test_ibvs_centering.py --live --drive      # also drives the wheels
        python tests/test_ibvs_centering.py --live --drive --arm  # + grabs when reached

    The ultrasonic sensor is polled on its own background thread
    (UltrasonicWatchdog, see ultrasonic_safety.py), not once per iteration
    of this loop -- it's the top-priority safety check, and camera
    inference plus the step-and-look time.sleep() below can each take
    longer than a single ultrasonic poll, during which a same-thread check
    would have gone blind. The watchdog calls actuator.stop() itself the
    instant it detects emergency_stop; this loop still reads its latest
    reading every iteration and checks emergency_stop first, before issuing
    any new drive command, so it can't immediately re-drive over the
    watchdog's stop.

    reactive_controller.py looks up the commanded speed from hardcoded
    distance breakpoints (CRUISE_DISTANCE_CM / FAR_DISTANCE_CM -> MAX_SPEED /
    APPROACH_SPEED / STEP_SPEED — see that module) rather than a continuous
    formula — same command either way, but HOW it's applied switches with
    distance: far out it's driven every frame like any other command
    (cruise/approach), while inside FAR_DISTANCE_CM (speed_tier() == "step")
    it's applied as a short STEP_DURATION_S nudge followed by a
    LOOK_PAUSE_S stop, so the camera gets a fresh, unblurred frame to re-aim
    from before the next nudge, walking the robot into the grab position
    instead of risking an overshoot on a stale frame. The overlay's DRIVE
    line and the console log both show reactive_controller.speed_tier()'s
    label (CRUISE / APPROACH / STEP / BACKUP / REACHED / FALLBACK / NONE)
    directly, so what's on screen can't drift out of sync with which
    constant is actually driving the wheels.

    If the can ends up too_close (bbox fills the frame), the command is a
    gentle BACKWARD nudge (BACKUP_SPEED) instead of a stop, so a bad
    distance calibration can't wedge the robot against the can; once
    backing off clears too_close, the tiers above re-approach and re-center
    on their own. Separately, if the can gets pushed closer while already
    at the "reached" handoff point, a short one-off backward nudge fires
    too (see last_distance_cm below) — the overlay flags that frame
    specifically instead of just showing a plain STOPPED.

    Keys (window focused): m = toggle drive, g = toggle arm, q = quit.
    Ctrl+C also stops.
    """
    import time
    import cv2
    import numpy as np
    from src.perception.detector import AluminiumCanDetector, RUNTIME_MODEL_PATH
    from src.hardware.actuators.pwm_driver import PWMActuator

    driving = "--drive" in sys.argv
    want_arm = "--arm" in sys.argv
    show_lines = "--line" in sys.argv
    controller_name = "reactive"

    cal = MotionCalibration()
    kin = DifferentialKinematics(cal)
    actuator = PWMActuator(calibration=cal)
    ultrasonic = UltrasonicSafety(trig=23, echo=24)

    # Defined before the watchdog starts: _retreat_pulse reads it from its thread.
    grab_state = {"cooldown_until": 0.0, "homed": False, "grabbing": False}

    def _retreat_pulse() -> None:
        """One bounded backward nudge, then stop. Stops before reversing per
        the H-bridge rule in docs/hardware_safety_patterns.md; bounded so the
        caller re-reads its sensor between pulses instead of reversing blind."""
        if grab_state["grabbing"]:
            # Only while the arm is mid-grab: the tin is then what the sensor
            # sees, and reversing would drag the chassis out from under it.
            return
        actuator.stop()
        time.sleep(0.05)
        actuator.apply(kin.backward(speed=BACKUP_SPEED))
        time.sleep(RETREAT_PULSE_S)
        actuator.stop()

    def _align_pulse(lateral_error: float) -> None:
        """One bounded pivot-in-place to recover nx alignment, then stop.

        Same sign convention as compute_reactive_command's arc_forward_*
        choice -- that mapping is hardware-specific and has flipped once
        before, so it is mirrored here rather than re-derived. Stops before
        turning for the same H-bridge reason as _retreat_pulse."""
        actuator.stop()
        time.sleep(0.05)
        actuator.apply(kin.turn_left(speed=ALIGN_TURN_SPEED) if lateral_error > 0
                       else kin.turn_right(speed=ALIGN_TURN_SPEED))
        time.sleep(ALIGN_PULSE_S)
        actuator.stop()

    # Polls the sensor in its own thread instead of once per main-loop
    # iteration, and retreats itself the instant it sees emergency_stop --
    # top priority, not delayed behind camera inference or a step-and-look
    # time.sleep() below. See UltrasonicWatchdog's docstring.
    ultra_watchdog = UltrasonicWatchdog(ultrasonic, stop_callback=_retreat_pulse).start()
    step_state = {"next_step_at": 0.0} # only consulted in reactive final-approach
    grab_latch = {"on": False, "until": 0.0}
    band = BAND_NOT_CALIBRATED   # last frame's band; gates the fast-path detect
    # Track the previous camera-based distance to detect someone pushing the
    # tin closer while we're already at the "reached" handoff point; if the
    # can moves noticeably closer, command a short backward nudge.
    last_distance_cm = None

    detector = AluminiumCanDetector(device="cpu", model_path=RUNTIME_MODEL_PATH,
                                     frame_width=1280, frame_height=720)
    detector.start()

    arm, solver, armed = None, None, False

    # A DRIVING dependency, not just an arm one -- it decides when to stop
    # closing in. --arm gates only the arm below.
    try:
        from src.arm.arc_grasp import ArcGraspSolver
        solver = ArcGraspSolver()
        if not solver.ready_for("upright") and not solver.ready_for("lying"):
            print(f"[arc] NOT calibrated ({solver.status()}) — band stop disabled.")
            solver = None
        else:
            print(f"[arc] {solver.status()}")
    except Exception as exc:
        print(f"[arc] solver unavailable ({exc}); band stop disabled.")

    if want_arm and solver is not None:
        try:
            from src.arm.grasp_planner import GraspPlanner
            arm = GraspPlanner(release_on_start=True)
            armed = True
            print("[arm] ready — press 'g' to toggle grabbing on/off.")
        except Exception as exc:
            print(f"[arm] not available ({exc}); continuing without it.")
    elif want_arm:
        print("[arm] disabled — arc_grasp not calibrated.")

    print(f"[live] drive={'ON' if driving else 'OFF'} arm={'ARMED' if armed else 'OFF'} "
          f"ctrl={controller_name} — m=toggle drive, g=toggle arm, q=quit")

    show = True
    try:
        while True:
            # Orientation only matters once the tin is near the arc band, and
            # it runs per detection per frame. Use LAST frame's band to skip
            # it while still cruising -- one stale frame at the boundary costs
            # nothing, since BAND_TOO_FAR can't solve a grab anyway.
            result = detector.detect(fast=(band == BAND_TOO_FAR))
            ultra = ultra_watchdog.latest   # background thread already stopped us if this is emergency_stop
            error = compute_target_error(result)

            cmd = compute_reactive_command(error, kin)
            tier = speed_tier(error)   # single source of truth for driving AND display below

            # Ask arc_grasp EVERY frame, and stop the moment it can solve --
            # not when the uncalibrated distance_cm says "reached".
            solved, tin_pose, point, band = _solve_for_detection(result, solver)
            if solved is not None:
                grab_latch["on"] = True
                grab_latch["until"] = time.monotonic() + GRABBABLE_LATCH_S
            elif grab_latch["on"] and time.monotonic() >= grab_latch["until"]:
                grab_latch["on"] = False
            # Latched: ny_tol_near is near-zero, so one frame of jitter drops
            # the tin out of the band. Without the latch that instantly
            # re-starts the approach -- straight back into the old loop.
            grabbable = grab_latch["on"]

            bbox_width_px = result.best.width if result.best is not None else None
            bbox_height_px = result.best.height if result.best is not None else None
            bbox_area_px = bbox_width_px * bbox_height_px if bbox_width_px is not None and bbox_height_px is not None else None

            recovered_nudge = False
            if driving:
                if grabbable:
                    # Ahead of emergency_stop: at grab range the ultrasonic is
                    # looking at the TARGET, and stopping is safe either way.
                    actuator.stop()

                elif ultra.emergency_stop:
                    pass   # watchdog thread owns retreat while this is true;
                           # don't fight it with a stop() every frame here

                elif band == BAND_TOO_CLOSE:
                    # Overshot past the NEAREST calibrated arc. Nothing else
                    # in the chain knows about the arc band, so without this
                    # the loop just keeps driving forward until the vision
                    # too_close guess or the ultrasonic catches it.
                    _retreat_pulse()

                elif band == BAND_NX_OUTSIDE and error.found:
                    # Inside the band's ny range but off the sampled nx span:
                    # a lateral miss. Arcing forward to fix it spends the
                    # remaining distance and lands in BAND_TOO_CLOSE, so turn
                    # in place instead and keep the distance we have.
                    _align_pulse(error.lateral_error)

                elif cmd is None:
                    actuator.stop()
                    # If we were 'reached' (arm handoff) but the can has been
                    # moved closer since the last frame, back off a bit even
                    # though the controller would normally return None.
                    if error.reached and error.distance_cm is not None and last_distance_cm is not None:
                        if error.distance_cm + 1.0 < last_distance_cm:
                            actuator.apply(kin.backward(speed=BACKUP_SPEED))
                            time.sleep(0.15)
                            actuator.stop()
                            recovered_nudge = True
                            print(f"[live] can moved closer (was {last_distance_cm:.1f}cm, "
                                  f"now {error.distance_cm:.1f}cm) -- backed off")

                elif error.too_close:
                    actuator.apply(kin.backward(speed=BACKUP_SPEED))

                elif tier == "step":
                    # Close range: a continuous command is already stale by
                    # the time it reaches the wheels, and stale matters more
                    # here. Nudge for one short step, then sit still long
                    # enough for the next frame to be a fresh, unblurred
                    # look before deciding the next step.
                    now = time.monotonic()
                    if now >= step_state["next_step_at"]:
                        actuator.apply(cmd)
                        time.sleep(STEP_DURATION_S)
                        actuator.stop()
                        step_state["next_step_at"] = time.monotonic() + LOOK_PAUSE_S
                    else:
                        actuator.stop()

                else:
                    actuator.apply(cmd) # cruise/approach: continuous, full-rate driving

            # solved is this frame's, not the latch: a stop may coast on a
            # stale solution, a grab may not. Keeps UltrasonicSafety.can_grab's
            # contract -- vision and ultrasonic stay two independent checks.
            if (armed and solved is not None
                    and ultra.grab_confirmed and not ultra.emergency_stop):
                grab_state["grabbing"] = True
                try:
                    _attempt_grab(solved, tin_pose, point, arm, actuator, grab_state)
                finally:
                    grab_state["grabbing"] = False
                grab_latch["on"] = False   # can should be gone -- re-evaluate fresh

            if show:
                try:
                    frame = detector.get_annotated_frame(result)
                    if frame is None:
                        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                        cv2.putText(frame, "[waiting for first frame...]", (20, 360),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                    if show_lines and solver is not None:
                        # Under the status banner so the banner stays readable.
                        _draw_arc_lines(frame, solver,
                                        tin_pose or "upright", point)
                    _draw_status_overlay(frame, error, cmd, driving, armed,
                                        bbox_area_px=bbox_area_px,
                                        controller_name=controller_name,
                                        tier=tier, recovered_nudge=recovered_nudge,
                                        ultra=ultra, solved=solved, grabbable=grabbable,
                                        band=band,
                                        applied=getattr(actuator, "last_duty", None))
                    cv2.imshow("IBVS centering  (m=drive g=arm l=lines []=floor -==step q=quit)",
                               frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if key == ord("m"):
                        driving = not driving
                        print(f"\n[mode] drive {'ON' if driving else 'OFF'}\n")
                        if not driving:
                            actuator.stop()
                    if key == ord("g") and arm is not None:
                        armed = not armed
                        print(f"\n[mode] arm {'ARMED' if armed else 'OFF'}\n")
                    if key == ord("l"):
                        show_lines = not show_lines
                        print(f"\n[mode] arc lines {'ON' if show_lines else 'OFF'}\n")
                    # Live tuning: both are read at call time (module globals),
                    # so mutating them here takes effect on the next command.
                    if key in (ord("["), ord("]")):
                        pwm_driver.MIN_MOVE_DUTY = max(
                            0.0, pwm_driver.MIN_MOVE_DUTY + (1.0 if key == ord("]") else -1.0))
                        print(f"[tune] MIN_MOVE_DUTY={pwm_driver.MIN_MOVE_DUTY:.0f}")
                    if key in (ord("-"), ord("=")):
                        rc.STEP_SPEED = max(
                            0.0, min(100.0, rc.STEP_SPEED + (1.0 if key == ord("=") else -1.0)))
                        print(f"[tune] STEP_SPEED={rc.STEP_SPEED:.0f}")
                except cv2.error:
                    print("[!] no display available — continuing text-only")
                    show = False

            print(f"[live] found={error.found} lateral={error.lateral_error:+.2f} "
                  f"dist_cm={error.distance_cm} last_dist_cm={last_distance_cm} "
                  f"bbox_width_px={bbox_width_px if result.best is not None else None} "
                  f"bbox_height_px={bbox_height_px if result.best is not None else None} "
                  f"bbox_area_px={bbox_area_px if bbox_width_px is not None and bbox_height_px is not None else None} "
                  f"ultra_dist={ultra.distance_cm} "
                  f"reached={error.reached} too_close={error.too_close} "
                  f"tier={tier} "
                  f"drive={'ON' if driving else 'OFF'} arm={'ARMED' if armed else 'OFF'} "
                  f"ctrl={controller_name}")

            # Update AFTER this frame's decisions/log use the previous value
            # -- see the "can moved closer" recovery above, which compares
            # against what was last seen, not the current frame.
            if error.distance_cm is not None:
                last_distance_cm = error.distance_cm
    except KeyboardInterrupt:
        pass
    finally:
        # Join the watchdog thread BEFORE closing the sensor/actuator it
        # touches -- otherwise a poll in flight could call actuator.stop()
        # or GPIO-read the sensor after either has been torn down.
        ultra_watchdog.stop()
        actuator.stop()
        actuator.close()
        detector.stop()
        ultrasonic.close()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    if "--live" in sys.argv:
        run_live_demo()
    else:
        _run_all_tests()
