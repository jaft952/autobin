"""
tests/test_ibvs_centering.py

Pure-logic tests for distance_error.py + approach_drive.py. NO hardware or
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
distance_error and the active controller (approach_drive by default, or
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
from src.visual_servoing.distance_error import compute_target_error
from src.visual_servoing.approach_drive import compute_drive_command, MAX_SPEED, BACKUP_SPEED
from src.visual_servoing.predictive_controller import ControllerState, compute_predictive_command


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


# ── approach_drive.compute_drive_command ─────────────────────────────────

def test_drive_command_none_when_not_found():
    error = compute_target_error(DetectionResult())
    assert compute_drive_command(error, _kin()) is None
    print("PASS drive command is None when nothing is found")


def test_drive_command_none_when_reached():
    error = compute_target_error(_make_detection(norm_x=0.5, bbox_height_px=300))
    assert error.reached
    assert not error.too_close, error
    assert compute_drive_command(error, _kin()) is None
    print("PASS drive command is None when reached (arm handoff)")


def test_drive_backs_away_when_too_close():
    """too_close -> an active BACKWARD command (both wheels negative), not
    just a stop — recovers from an overshoot instead of sitting wedged
    against the can, even when off-center."""
    error = compute_target_error(_make_detection(norm_x=0.3, bbox_height_px=360))
    assert error.too_close
    cmd = compute_drive_command(error, _kin())
    assert cmd is not None
    assert cmd.left_speed < 0 and cmd.right_speed < 0, cmd
    assert abs(cmd.left_speed) <= BACKUP_SPEED + 1e-6, cmd
    print("PASS backs away (both wheels reverse) when too_close")


def test_drive_when_can_is_left_of_center():
    """Far + left of center -> steers toward arc_forward_right (right wheel
    slower than left) — hardware-verified mapping, see the sign-convention
    note in approach_drive.py. Far away -> speed at MAX_SPEED (the ramp only
    starts falling off inside FAR_DISTANCE_CM)."""
    error = compute_target_error(_make_detection(norm_x=0.3, bbox_height_px=40))
    cmd = compute_drive_command(error, _kin())
    assert cmd is not None
    assert cmd.right_speed < cmd.left_speed, cmd
    assert cmd.left_speed >= 0.9 * MAX_SPEED, cmd   # brisk, not crawling
    print("PASS steers toward the can + full speed when far + left of center")


def test_drive_when_can_is_right_of_center():
    """Close + right of center -> steers toward arc_forward_left (left wheel
    slower than right) — hardware-verified mapping, see approach_drive.py.
    Close to STOP_DISTANCE_CM -> speed ramped down near zero, not a fixed
    floor (no more pulsing: slow continuous creep IS the final-approach
    behavior now)."""
    error = compute_target_error(_make_detection(norm_x=0.7, bbox_height_px=200))
    cmd = compute_drive_command(error, _kin())
    assert cmd is not None
    assert cmd.left_speed < cmd.right_speed, cmd
    assert 0 <= cmd.right_speed < 0.3 * MAX_SPEED, cmd   # ramped down close to zero
    print("PASS steers toward the can + slow creep when close + right of center")


def test_speed_ramps_down_as_distance_shrinks():
    """Same lateral offset, three distances -> speed strictly decreases as
    the can gets closer, confirming the continuous ramp (not a floor/step)."""
    far = compute_drive_command(
        compute_target_error(_make_detection(norm_x=0.7, bbox_height_px=40)), _kin())    # ~100cm
    mid = compute_drive_command(
        compute_target_error(_make_detection(norm_x=0.7, bbox_height_px=100)), _kin())   # 40cm
    near = compute_drive_command(
        compute_target_error(_make_detection(norm_x=0.7, bbox_height_px=200)), _kin())   # 20cm
    assert far is not None and mid is not None and near is not None
    assert far.left_speed > mid.left_speed > near.left_speed, (far, mid, near)
    print("PASS speed ramps down continuously as distance shrinks (no pulsing)")


# ── Plain runner (no pytest needed) ───────────────────────────────────────

ALL_TESTS = [
    test_no_detection_not_found,
    test_far_left_of_center,
    test_close_right_of_center,
    test_reached_when_centered_and_close,
    test_too_close_backs_away_even_when_off_center,
    test_drive_command_none_when_not_found,
    test_drive_command_none_when_reached,
    test_drive_backs_away_when_too_close,
    test_drive_when_can_is_left_of_center,
    test_drive_when_can_is_right_of_center,
    test_speed_ramps_down_as_distance_shrinks,
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


def _draw_status_overlay(frame, error, cmd, driving: bool, armed: bool,
                         bbox_height_px=None, controller_name: str = "reactive") -> None:
    """Burn the distance_error / drive-command readout onto the frame so the
    planned IK output (steer + dynamic speed) is visible without reading the
    console, and so a bad estimate is obvious immediately.

    bbox_height_px is shown raw (not just the derived distance_cm) because
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
        bbox_txt = f"{bbox_height_px}px" if bbox_height_px is not None else "?"
        target_line = (f"TARGET: lateral={error.lateral_error:+.2f} "
                        f"dist_cm={dist_txt} bbox_h={bbox_txt} "
                        f"reached={error.reached} too_close={error.too_close}")
        target_color = (0, 255, 0) if error.reached else \
            ((0, 140, 255) if error.too_close else (0, 255, 255))
    cv2.putText(frame, target_line, (16, fh - banner_h + 26), font, 0.55, target_color, 2)

    if cmd is None:
        reason = "reached" if error.reached else ("no target" if not error.found else "stopped")
        drive_line, drive_color = f"DRIVE: STOPPED ({reason})", (170, 170, 170)
    elif cmd.left_speed < 0 and cmd.right_speed < 0:
        drive_line = f"DRIVE: BACKING UP (too close) L={cmd.left_speed:+.1f} R={cmd.right_speed:+.1f}"
        drive_color = (0, 140, 255)
    else:
        # Derived from lateral_error's sign (which way the CAN is, and so
        # which way we're correcting), not from comparing wheel speeds —
        # that comparison depends on which kin method approach_drive.py
        # happens to call for a given sign, which is hardware-specific and
        # has already flipped once (see the note in approach_drive.py).
        if error.lateral_error > 0.02:
            steer = "RIGHT"
        elif error.lateral_error < -0.02:
            steer = "LEFT"
        else:
            steer = "STRAIGHT"
        drive_line = f"DRIVE: L={cmd.left_speed:+.1f} R={cmd.right_speed:+.1f} (steer={steer})"
        drive_color = (0, 255, 0)
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
    approach_drive readout burned onto the frame, so the planned IK output
    can be checked visually before it ever touches the motors.

    Run by hand on the Pi:
        python tests/test_ibvs_centering.py --live              # look only, nothing moves
        python tests/test_ibvs_centering.py --live --drive      # also drives the wheels
        python tests/test_ibvs_centering.py --live --drive --arm  # + grabs when reached

    No separate pulse/cruise mode: approach_drive.py ramps the commanded
    speed continuously from MAX_SPEED down to 0 as the can's estimated
    distance closes in on STOP_DISTANCE_CM, so it's just applied every frame
    like any other command — fast far away, a crawl right at the grab spot.
    If the can ends up too_close (bbox fills the frame), the command is an
    active BACKWARD nudge instead of a stop, so a bad distance calibration
    can't wedge the robot against the can; once backing off clears
    too_close, the same ramp re-approaches and re-centers on its own.

    Keys (window focused): m = toggle drive, g = toggle arm, q = quit.
    Ctrl+C also stops.
    """
    import cv2
    import numpy as np
    from src.perception.detector import AluminiumCanDetector, RUNTIME_MODEL_PATH
    from src.hardware.actuators.pwm_driver import PWMActuator

    driving = "--drive" in sys.argv
    want_arm = "--arm" in sys.argv
    use_predictive = "--predictive" in sys.argv
    controller_name = "predictive" if use_predictive else "reactive"

    cal = MotionCalibration()
    kin = DifferentialKinematics(cal)
    actuator = PWMActuator(calibration=cal)
    controller_state = ControllerState()  # only consulted when --predictive is set

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
            error = compute_target_error(result)
            if use_predictive:
                cmd, controller_state = compute_predictive_command(error, kin, controller_state)
            else:
                cmd = compute_drive_command(error, kin)
            bbox_height_px = result.best.height if result.best is not None else None

            if driving:
                actuator.apply(cmd) if cmd is not None else actuator.stop()

            if armed and error.reached:
                _attempt_grab(result, solver, arm, actuator, grab_state)

            if show:
                try:
                    frame = detector.get_annotated_frame(result)
                    if frame is None:
                        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                        cv2.putText(frame, "[waiting for first frame...]", (20, 360),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                    _draw_status_overlay(frame, error, cmd, driving, armed,
                                        bbox_height_px=bbox_height_px,
                                        controller_name=controller_name)
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

            print(f"[live] found={error.found} lateral={error.lateral_error:+.2f} "
                  f"dist_cm={error.distance_cm} bbox_h_px={bbox_height_px} "
                  f"reached={error.reached} too_close={error.too_close} "
                  f"drive={'ON' if driving else 'OFF'} arm={'ARMED' if armed else 'OFF'} "
                  f"ctrl={controller_name}")
    except KeyboardInterrupt:
        pass
    finally:
        actuator.stop()
        actuator.close()
        detector.stop()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    if "--live" in sys.argv:
        run_live_demo()
    else:
        _run_all_tests()
