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
from src.visual_servoing.distance_error import compute_target_error
from src.visual_servoing.reactive_controller import (
    compute_reactive_command, is_final_approach,
    MAX_SPEED, BACKUP_SPEED, STEP_DURATION_S, LOOK_PAUSE_S,
)
# predictive approach removed — only reactive controller used
from src.visual_servoing.ultrasonic_safety import UltrasonicSafety


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
    note in reactive_controller.py. Far away -> speed at MAX_SPEED (the ramp only
    starts falling off inside FAR_DISTANCE_CM)."""
    error = compute_target_error(_make_detection(norm_x=0.3, bbox_height_px=40))
    cmd = compute_reactive_command(error, _kin())
    assert cmd is not None
    assert cmd.right_speed < cmd.left_speed, cmd
    assert cmd.left_speed >= 0.9 * MAX_SPEED, cmd   # brisk, not crawling
    print("PASS steers toward the can + full speed when far + left of center")


def test_reactive_command_when_can_is_right_of_center():
    """Close + right of center -> steers toward arc_forward_left (left wheel
    slower than right) — hardware-verified mapping, see reactive_controller.py.
    Close to STOP_DISTANCE_CM -> speed ramped down near zero, not a fixed
    floor (no more pulsing: slow continuous creep IS the final-approach
    behavior now)."""
    error = compute_target_error(_make_detection(norm_x=0.7, bbox_height_px=200))
    cmd = compute_reactive_command(error, _kin())
    assert cmd is not None
    assert cmd.left_speed < cmd.right_speed, cmd
    assert 0 <= cmd.right_speed < 0.3 * MAX_SPEED, cmd   # ramped down close to zero
    print("PASS steers toward the can + slow creep when close + right of center")


def test_speed_ramps_down_as_distance_shrinks():
    """Same lateral offset, three distances -> speed strictly decreases as
    the can gets closer, confirming the continuous ramp (not a floor/step)."""
    far = compute_reactive_command(
        compute_target_error(_make_detection(norm_x=0.7, bbox_height_px=40)), _kin())    # ~100cm
    mid = compute_reactive_command(
        compute_target_error(_make_detection(norm_x=0.7, bbox_height_px=100)), _kin())   # 40cm
    near = compute_reactive_command(
        compute_target_error(_make_detection(norm_x=0.7, bbox_height_px=200)), _kin())   # 20cm
    assert far is not None and mid is not None and near is not None
    assert far.left_speed > mid.left_speed > near.left_speed, (far, mid, near)
    print("PASS speed ramps down continuously as distance shrinks (no pulsing)")


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
    test_speed_ramps_down_as_distance_shrinks,
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
                         bbox_area_px=None, controller_name: str = "reactive",
                         final_approach: bool = False) -> None:
    """Burn the distance_error / drive-command readout onto the frame so the
    planned IK output (steer + dynamic speed) is visible without reading the
    console, and so a bad estimate is obvious immediately.

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

    if cmd is None:
        reason = "reached" if error.reached else ("no target" if not error.found else "stopped")
        drive_line, drive_color = f"DRIVE: STOPPED ({reason})", (170, 170, 170)
    elif cmd.left_speed < 0 and cmd.right_speed < 0:
        drive_line = f"DRIVE: BACKING UP (too close) L={cmd.left_speed:+.1f} R={cmd.right_speed:+.1f}"
        drive_color = (0, 140, 255)
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
        mode = "STEP" if final_approach else "CRUISE"
        drive_line = (f"DRIVE [{mode}]: L={cmd.left_speed:+.1f} R={cmd.right_speed:+.1f} "
                      f"(steer={steer})")
        drive_color = (0, 200, 255) if final_approach else (0, 255, 0)
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

    reactive_controller.py ramps the commanded speed continuously from
    MAX_SPEED down to 0 as the can's estimated distance closes in on
    STOP_DISTANCE_CM — same command either way, but HOW it's applied
    switches with distance (reactive mode only; --predictive drives its own
    planned command continuously throughout): far out it's driven every
    frame like any other command (cruise), while inside FAR_DISTANCE_CM
    (is_final_approach()) it's applied as a short STEP_DURATION_S nudge
    followed by a LOOK_PAUSE_S stop, so the camera gets a fresh, unblurred
    frame to re-aim from before the next nudge, walking the robot into the
    grab position instead of risking an overshoot on a stale frame. The
    overlay's DRIVE line shows [CRUISE] or [STEP].

    If the can ends up too_close (bbox fills the frame), the command is an
    active BACKWARD nudge instead of a stop, so a bad distance calibration
    can't wedge the robot against the can; once backing off clears
    too_close, the same ramp re-approaches and re-centers on its own.

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
    step_state = {"next_step_at": 0.0}    # only consulted in reactive final-approach

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
            ultra = ultrasonic.update()
            error = compute_target_error(result)
            

            cmd = compute_reactive_command(error, kin)
            final_approach = is_final_approach(error)

            bbox_width_px = result.best.width if result.best is not None else None
            bbox_height_px = result.best.height if result.best is not None else None
            bbox_area_px = bbox_width_px * bbox_height_px if bbox_width_px is not None and bbox_height_px is not None else None

            if driving:
                if ultra.emergency_stop:
                    actuator.stop()

                elif cmd is None:
                    actuator.stop()

                elif final_approach:
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
                    actuator.apply(cmd)   # cruise: continuous, full-rate driving

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
                                        final_approach=final_approach)
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
                  f"dist_cm={error.distance_cm} "
                  f"bbox_width_px={bbox_width_px if result.best is not None else None} "
                  f"bbox_height_px={bbox_height_px if result.best is not None else None} "
                  f"bbox_area_px={bbox_area_px if bbox_width_px is not None and bbox_height_px is not None else None} "
                  f"ultra_dist={ultra.distance_cm} "
                  f"reached={error.reached} too_close={error.too_close} "
                  f"phase={'STEP' if final_approach else 'CRUISE'} "
                  f"drive={'ON' if driving else 'OFF'} arm={'ARMED' if armed else 'OFF'} "
                  f"ctrl={controller_name}")
    except KeyboardInterrupt:
        pass
    finally:
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
