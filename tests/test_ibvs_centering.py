"""
tests/test_ibvs_centering.py

Pure-logic tests for distance_error.py + reactive_controller.py + target_lock.py
(see src/perception/target_lock.py -- picks the largest-bbox tin and holds it
across frames, one at a time, instead of re-deriving "the" target from
DetectionResult.best every frame). NO hardware or camera needed — builds
synthetic DetectionResult/BoundingBox fixtures by hand, same spirit as
tests/test_scan_logic.py:

    python tests/test_ibvs_centering.py      # plain runner with PASS/FAIL summary
    pytest tests/test_ibvs_centering.py      # also works
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.perception.detector import BoundingBox, DetectionResult
from src.motion.calibration import MotionCalibration
from src.motion.differential_kinematics import DifferentialKinematics
from src.visual_servoing.distance_error import compute_target_error, TargetError
from src.visual_servoing.reactive_controller import (
    compute_reactive_command, speed_tier,
    FORWARD_HIGH_SPEED, FORWARD_MID_SPEED, FORWARD_LOW_SPEED, BACKUP_SPEED,
)
from src.perception.target_lock import TargetLock


# ── Fixture builder ──────────────────────────────────────────────────────

def _make_box(norm_x: float, bbox_height_px: float, confidence: float = 0.9,
              bbox_width_px: float = 60, frame_width: int = 640) -> BoundingBox:
    """A single BoundingBox whose ground-contact point sits at norm_x (0..1
    across the frame) with the given pixel bbox height (drives the monocular
    distance estimate) and width (drives bbox area)."""
    center_x = norm_x * frame_width
    half_w = bbox_width_px / 2
    x1 = int(center_x - half_w)
    x2 = int(center_x + half_w)
    y1 = 100
    y2 = int(y1 + bbox_height_px)
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, confidence=confidence)


def _make_detection(norm_x: float, bbox_height_px: float,
                     frame_width: int = 640, frame_height: int = 480) -> DetectionResult:
    """A single detection whose ground-contact point sits at norm_x (0..1
    across the frame) with the given pixel bbox height (drives the monocular
    distance estimate)."""
    box = _make_box(norm_x, bbox_height_px, frame_width=frame_width)
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
    note in reactive_controller.py. Far away (> FAR_DISTANCE_CM) ->
    speed at FORWARD_HIGH_SPEED, the top hardcoded tier."""
    error = compute_target_error(_make_detection(norm_x=0.3, bbox_height_px=40))
    cmd = compute_reactive_command(error, _kin())
    assert cmd is not None
    assert cmd.right_speed < cmd.left_speed, cmd
    assert cmd.left_speed >= 0.9 * FORWARD_HIGH_SPEED, cmd   # brisk, not crawling
    print("PASS steers toward the can + full speed when far + left of center")


def test_reactive_command_speed_tiers_are_hardcoded():
    """Same lateral offset, three distances spanning the three breakpoints
    -> each picks a distinct hardcoded tier (per speed_tier()'s label) with
    the outer wheel's speed matching that tier's constant, confirming a
    lookup against LOW_DISTANCE_CM / FAR_DISTANCE_CM rather than a
    continuous formula."""
    cruise_error = compute_target_error(_make_detection(norm_x=0.7, bbox_height_px=40))                        # ~142cm
    approach_error = compute_target_error(_make_detection(norm_x=0.7, bbox_height_px=400, frame_height=5000))  # ~45cm
    step_error = compute_target_error(_make_detection(norm_x=0.7, bbox_height_px=1064, frame_height=5000))     # ~27cm

    assert speed_tier(cruise_error) == "cruise", cruise_error
    assert speed_tier(approach_error) == "approach", approach_error
    assert speed_tier(step_error) == "step", step_error

    cruise = compute_reactive_command(cruise_error, _kin())
    approach = compute_reactive_command(approach_error, _kin())
    step = compute_reactive_command(step_error, _kin())
    assert cruise is not None and approach is not None and step is not None
    assert abs(cruise.right_speed - FORWARD_HIGH_SPEED) < 0.2 * FORWARD_HIGH_SPEED, cruise
    assert abs(approach.right_speed - FORWARD_MID_SPEED) < 0.2 * FORWARD_MID_SPEED, approach
    assert abs(step.right_speed - FORWARD_LOW_SPEED) < 0.2 * FORWARD_LOW_SPEED, step
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
        _make_detection(norm_x=0.3, bbox_height_px=400, frame_height=5000))) == "approach"   # ~45cm
    assert speed_tier(compute_target_error(
        _make_detection(norm_x=0.5, bbox_height_px=1064, frame_height=5000))) == "step"   # ~27cm
    fallback_error = TargetError(found=True, lateral_error=0.0, distance_cm=None,
                                  reached=False, too_close=False)
    assert speed_tier(fallback_error) == "fallback"
    print("PASS speed_tier labels match every compute_reactive_command branch")


# ── TargetLock prioritization ────────────────────────────────────────────
#
# The live loop used to feed every frame's DetectionResult.best straight into
# compute_target_error / _solve_for_detection -- max CONFIDENCE, recomputed
# independently every frame with no memory. With two tins in view that
# flips which one "best" points at frame to frame (confidence jitters,
# whichever detection happens to be biggest that instant wins), so the robot
# would drift off a can it was mid-approach on instead of finishing the
# grab. These tests drive TargetLock the same way run_live_demo() does: feed
# it raw detections, wrap the box it returns in a single-detection
# DetectionResult, and check THAT is what steering/distance actually locks
# onto -- largest bbox area first, held until released (grab done).


def test_locks_onto_largest_bbox_not_highest_confidence():
    """Two tins in one frame: a physically bigger, lower-confidence one and
    a smaller, higher-confidence one. TargetLock must pick the bigger box --
    confidence never enters the decision."""
    small_confident = _make_box(norm_x=0.2, bbox_height_px=40, confidence=0.99, bbox_width_px=30)
    big_unsure = _make_box(norm_x=0.7, bbox_height_px=200, confidence=0.75, bbox_width_px=150)
    lock = TargetLock()
    picked = lock.select([small_confident, big_unsure], now=0.0)
    assert picked is big_unsure, "largest bbox area must win, not highest confidence"
    print("PASS TargetLock picks the largest bbox area over the more confident detection")


def test_reactive_command_targets_the_locked_can_not_a_bigger_newcomer():
    """Approaching the locked (initially largest) can when a second, now-
    bigger can appears nearby -- the drive command must still steer toward
    the ORIGINAL locked can's position, not jump to the newcomer. Mirrors
    run_live_demo(): TargetLock.select() output feeds a single-detection
    DetectionResult into compute_target_error, same as the live loop does."""
    tin = _make_box(norm_x=0.3, bbox_height_px=150, bbox_width_px=100)
    lock = TargetLock()
    assert lock.select([tin], now=0.0) is tin

    newcomer = _make_box(norm_x=0.7, bbox_height_px=250, bbox_width_px=200)  # bigger, elsewhere
    locked = lock.select([tin, newcomer], now=0.2)
    assert locked is not None and locked is tin, "must stay locked on the original can"

    locked_result = DetectionResult(detections=[locked], frame_width=640, frame_height=480)
    error = compute_target_error(locked_result)
    cmd = compute_reactive_command(error, _kin())
    assert error.lateral_error < 0, "locked can is left of center -> negative error"
    assert cmd is not None and cmd.right_speed < cmd.left_speed, cmd
    print("PASS drive command steers toward the locked can, ignoring the bigger newcomer")


def test_advances_to_next_largest_after_release():
    """release() is what run_live_demo() calls right after a successful
    grab -- the freed-up tin is gone from the frame, so the very next
    select() must pick the largest of what's LEFT, moving the robot on to
    the next target instead of re-locking the (now collected) one."""
    first = _make_box(norm_x=0.5, bbox_height_px=300, bbox_width_px=250)
    second = _make_box(norm_x=0.2, bbox_height_px=150, bbox_width_px=100)
    lock = TargetLock()
    assert lock.select([first, second], now=0.0) is first

    lock.release()   # simulates _attempt_grab() succeeding on `first`
    # `first` is gone (collected); only `second` remains in view.
    picked = lock.select([second], now=0.1)
    assert picked is second, "must advance to the next-largest remaining tin"
    print("PASS releasing the lock after a grab advances to the next-largest tin")


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
    test_reactive_command_speed_tiers_are_hardcoded,
    test_speed_tier_labels_cover_every_branch,
    test_locks_onto_largest_bbox_not_highest_confidence,
    test_reactive_command_targets_the_locked_can_not_a_bigger_newcomer,
    test_advances_to_next_largest_after_release,
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


if __name__ == "__main__":
    _run_all_tests()
