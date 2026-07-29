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
    python tests/test_ibvs_centering.py --live --drive --predictive  # use the receding-horizon
                                                                       # planner instead of the
                                                                       # reactive controller

--drive, --arm and --predictive are ONLY read together with --live; they do
nothing to the pure-logic test suite. Without --drive the wheels never move —
distance_error and the active controller (reactive_controller by default, or
predictive_controller with --predictive) still run every frame and their
output is shown on the video overlay and console, so you can check the
planned steer/speed before ever letting it touch the motors. Keys while the
window is focused: m = toggle drive, g = toggle arm, q = quit.

This whole function is NOT executed by the test suite or by pytest — see
run_live_demo() at the bottom.
"""
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))   # for test_arc_grasp (--arm)

from src.perception.detector import BoundingBox, DetectionResult
from src.motion.calibration import MotionCalibration
from src.motion.differential_kinematics import DifferentialKinematics
from src.visual_servoing.distance_error import compute_target_error, TargetError
from src.visual_servoing.reactive_controller import (
    compute_reactive_command, speed_tier,
    FORWARD_HIGH_SPEED, FORWARD_MID_SPEED, FORWARD_LOW_SPEED, BACKUP_SPEED,
)
# predictive approach removed — only reactive controller used
from src.visual_servoing.ultrasonic_safety import UltrasonicSafety, UltrasonicWatchdog


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

_GRAB_COOLDOWN_S = 3.0   # after a grab attempt (success or refusal), don't re-solve every frame

# TODO: reactive_controller.py dropped STEP_DURATION_S / LOOK_PAUSE_S (its own
# docstring says step-and-look timing is "the caller's job... not here") but
# doesn't export replacements -- these are the old values migrated here as a
# placeholder purely so run_live_demo()'s step-and-look branch doesn't throw
# NameError. Tune or move these wherever you decide they belong.
STEP_DURATION_S = 0.25
LOOK_PAUSE_S = 0.6


def _pose_and_angle(box):
    """(pose, angle_deg_or_None) for arc_grasp, from the detection's
    segmentation-based orientation — same convention as src/arm/arc_grasp.py
    and the old ibvs_centering pipeline. No usable mask -> assume upright."""
    o = box.orientation
    if o is None or o.klass == "upright":
        return "upright", None
    if o.klass == "axial":
        return "lying", None
    return "lying", o.angle


TIER_COLOR = {
    "cruise": (0, 255, 0),      # green      -- brisk, FORWARD_HIGH_SPEED
    "approach": (0, 220, 170),  # teal       -- FORWARD_MID_SPEED
    "step": (0, 200, 255),      # orange     -- FORWARD_LOW_SPEED, step-and-look nudge
    "backup": (0, 140, 255),    # red-orange -- BACKUP_SPEED, too_close
    "reached": (0, 255, 0),     # green      -- arm handoff point
    "fallback": (0, 165, 255),  # amber      -- FORWARD_LOW_SPEED, distance unknown
    "none": (0, 0, 255),        # red        -- nothing found
}


def _format_motion_plan(cmd, error, tier: str, ultra_emergency: bool = False) -> str:
    """Format the planned motion for clear console display."""
    if ultra_emergency:
        return "ULTRASONIC OVERRIDE: EMERGENCY STOP"
    if cmd is None:
        return f"STOPPED ({tier})"
    if cmd.left_speed < 0 and cmd.right_speed < 0:
        return f"BACKING UP: L={cmd.left_speed:+.1f} R={cmd.right_speed:+.1f}"

    steer = "RIGHT" if error.lateral_error > 0.02 else "LEFT" if error.lateral_error < -0.02 else "STRAIGHT"

    if steer == "LEFT":
        expected = "L_faster" if cmd.left_speed > cmd.right_speed else "R_faster"
        status = "[OK]" if cmd.left_speed > cmd.right_speed else "[ERR]"
    elif steer == "RIGHT":
        expected = "R_faster" if cmd.right_speed > cmd.left_speed else "L_faster"
        status = "[OK]" if cmd.right_speed > cmd.left_speed else "[ERR]"
    else:
        expected = "equal"
        status = "[OK]" if abs(cmd.left_speed - cmd.right_speed) < 0.5 else "[WARN]"

    return f"{tier.upper()}: {steer} {status} | L={cmd.left_speed:+.1f} R={cmd.right_speed:+.1f} ({expected})"


def _draw_status_overlay(frame, error, cmd, driving: bool, armed: bool,
                         bbox_area_px=None, controller_name: str = "reactive",
                         tier: str = "none", recovered_nudge: bool = False,
                         ultra_emergency: bool = False) -> None:
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
    calibrate it (see distance_error.py)."""
    import cv2
    fh, fw = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    banner_h = 110
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

    if ultra_emergency:
        drive_line = "DRIVE: [ULTRASONIC] EMERGENCY STOP"
        drive_color = (0, 0, 255) # red
    elif cmd is None:
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
    cv2.putText(frame, drive_line, (16, fh - banner_h + 54), font, 0.6, drive_color, 2)

    mode_line = (f"[m] drive={'ON' if driving else 'OFF'}   [g] arm={'ARMED' if armed else 'OFF'}   "
                 f"ctrl={controller_name}   [q] quit")
    cv2.putText(frame, mode_line, (16, fh - banner_h + 80), font, 0.5, (200, 200, 200), 1)

    dcol = (0, 255, 0) if driving else (0, 0, 255)
    cv2.putText(frame, "DRIVE ON" if driving else "DRIVE OFF", (fw - 190, 30), font, 0.65, (0, 0, 0), 3)
    cv2.putText(frame, "DRIVE ON" if driving else "DRIVE OFF", (fw - 190, 30), font, 0.65, dcol, 1)
    if armed:
        cv2.putText(frame, "ARM ARMED", (fw - 190, 58), font, 0.65, (0, 0, 0), 3)
        cv2.putText(frame, "ARM ARMED", (fw - 190, 58), font, 0.65, (0, 255, 0), 1)


def _attempt_grab(result: DetectionResult, solver, arm, actuator, state: dict) -> None:
    """Solve an arc-grasp pose from the current detection and, if grabbable,
    run the SAME grab -> dump -> home sequence tests/test_arc_grasp.py uses
    (via the shared Arm class), so this is byte-for-byte the grab you tune
    there. Cooldown-gated so a tin sitting in the band doesn't re-trigger
    every single frame."""
    import time as _time
    if _time.monotonic() < state["cooldown_until"]:
        return
    best = result.best
    if best is None:
        return
    pose, angle = _pose_and_angle(best)
    point = result.normalized_center() if pose == "lying" else result.normalized_base_center()
    if point is None:
        return

    solved = solver.solve(point[0], point[1], pose=pose, angle_deg=angle)
    state["cooldown_until"] = _time.monotonic() + _GRAB_COOLDOWN_S
    if solved is None:
        print(f"[arm] {pose} tin at nx={point[0]:.2f} ny={point[1]:.2f} "
              f"— not in the calibrated grab band ({solver.status()})")
        return

    print(f"[arm] GRAB ({pose}) @ nx={point[0]:.2f} ny={point[1]:.2f}")
    if not state["homed"]:
        arm.force_home()   # tracked pose may be from a stale prior session
        state["homed"] = True
    arm.grab(solved, tin_pose=pose)   # brakes the wheels internally
    arm.dump_to_bin()
    arm.rest()
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
    distance breakpoints (FAR_DISTANCE_CM / LOW_DISTANCE_CM -> FORWARD_HIGH_SPEED /
    FORWARD_MID_SPEED / FORWARD_LOW_SPEED — see that module) rather than a continuous
    formula — same command either way, but HOW it's applied switches with
    distance: far out it's driven every frame like any other command
    (cruise/approach), while inside LOW_DISTANCE_CM (speed_tier() == "step")
    it's applied as a short STEP_DURATION_S nudge followed by a
    LOOK_PAUSE_S stop (TODO: reactive_controller.py no longer defines either
    constant -- see the placeholders defined near _GRAB_COOLDOWN_S below),
    so the camera gets a fresh, unblurred frame to re-aim
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
    controller_name = "reactive"

    cal = MotionCalibration()
    kin = DifferentialKinematics(cal)
    actuator = PWMActuator(calibration=cal)
    ultrasonic = UltrasonicSafety(trig=23, echo=24)
    # Polls the sensor in its own thread instead of once per main-loop
    # iteration, and calls actuator.stop() itself the instant it sees
    # emergency_stop -- top priority, not delayed behind camera inference or
    # a step-and-look time.sleep() below. See UltrasonicWatchdog's docstring.
    ultra_watchdog = UltrasonicWatchdog(ultrasonic, stop_callback=actuator.stop).start()
    step_state = {"next_step_at": 0.0} # only consulted in the "step" speed tier
    # Track the previous camera-based distance to detect someone pushing the
    # tin closer while we're already at the "reached" handoff point; if the
    # can moves noticeably closer, command a short backward nudge.
    last_distance_cm = None

    detector = AluminiumCanDetector(device="cpu", model_path=RUNTIME_MODEL_PATH,
                                     frame_width=1280, frame_height=720)
    detector.start()

    arm, solver, armed = None, None, False
    grab_state = {"cooldown_until": 0.0, "homed": False}
    if want_arm:
        try:
            from src.arm.arc_grasp import ArcGraspSolver
            from test_arc_grasp import Arm   # tests/ is on sys.path (see top of file)
            solver = ArcGraspSolver()
            if not solver.ready_for("upright") and not solver.ready_for("lying"):
                print(f"[arm] arc_grasp not calibrated ({solver.status()}) — arm disabled.")
            else:
                # Share OUR actuator: Arm() building its own would GPIO.cleanup
                # the shared motor pins and kill this loop's wheel control.
                arm = Arm(wheels=actuator)
                armed = True
                print("[arm] ready — press 'g' to toggle grabbing on/off.")
        except Exception as exc:
            print(f"[arm] not available ({exc}); continuing without it.")

    print(f"[live] drive={'ON' if driving else 'OFF'} arm={'ARMED' if armed else 'OFF'} "
          f"ctrl={controller_name} — m=toggle drive, g=toggle arm, q=quit")

    show = True
    try:
        while True:
            result = detector.detect()
            ultra = ultra_watchdog.latest   # background thread already stopped us if this is emergency_stop
            error = compute_target_error(result)

            cmd = compute_reactive_command(error, kin)
            tier = speed_tier(error)   # single source of truth for driving AND display below

            bbox_width_px = result.best.width if result.best is not None else None
            bbox_height_px = result.best.height if result.best is not None else None
            bbox_area_px = bbox_width_px * bbox_height_px if bbox_width_px is not None and bbox_height_px is not None else None

            recovered_nudge = False
            if driving:
                if ultra.emergency_stop:
                    actuator.stop()
                    print(f"  [SAFETY] Ultrasonic emergency stop triggered at {ultra.distance_cm}cm!")

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

            if armed and error.reached and ultra.grab_confirmed and not ultra.emergency_stop:
                _attempt_grab(result, solver, arm, actuator, grab_state)

            if show:
                try:
                    frame = detector.get_annotated_frame(result)
                    if frame is None:
                        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                        cv2.putText(frame, "[waiting for first frame...]", (20, 360),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                    _draw_status_overlay(frame, error, cmd, driving, armed,
                                        bbox_area_px=bbox_area_px,
                                        controller_name=controller_name,
                                        tier=tier, recovered_nudge=recovered_nudge,
                                        ultra_emergency=ultra.emergency_stop)
                    cv2.imshow("IBVS centering  (m=drive g=arm q=quit)", frame)
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
                except cv2.error:
                    print("[!] no display available — continuing text-only")
                    show = False

            motion_plan = _format_motion_plan(cmd, error, tier, ultra.emergency_stop)
            ultra_status = f"ULTRASONIC: {ultra.distance_cm}cm | emergency={ultra.emergency_stop} grab_ok={ultra.grab_confirmed}"
            target_status = f"TARGET: found={error.found} lateral={error.lateral_error:+.2f} dist={error.distance_cm}cm reached={error.reached} too_close={error.too_close}"
            bbox_info = f"BBOX: w={bbox_width_px} h={bbox_height_px} area={bbox_area_px}px" if bbox_area_px else "BBOX: none"
            mode_status = f"MODE: drive={'ON' if driving else 'OFF'} arm={'ARMED' if armed else 'OFF'} ctrl={controller_name}"

            print("\n" + "-" * 80)
            print(f"[MOTION] {motion_plan}")
            print(f"[ULTRASONIC] {ultra_status}")
            print(f"[TARGET] {target_status}")
            print(f"[BBOX] {bbox_info}")
            print(f"[MODE] {mode_status}")

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
