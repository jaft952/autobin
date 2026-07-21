"""
tests/test_ibvs_centering.py

Pure-logic tests for distance_error.py + approach_drive.py. NO hardware or
camera needed — builds synthetic DetectionResult/BoundingBox fixtures by
hand, same spirit as tests/test_scan_logic.py:

    python tests/test_ibvs_centering.py      # plain runner with PASS/FAIL summary
    pytest tests/test_ibvs_centering.py      # also works

For a live run on the robot (real camera + motors) use:

    python tests/test_ibvs_centering.py --live

which is NOT executed by the test suite or by pytest — see run_live_demo()
at the bottom.
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.perception.detector import BoundingBox, DetectionResult
from src.motion.calibration import MotionCalibration
from src.motion.differential_kinematics import DifferentialKinematics
from src.visual_servoing.distance_error import compute_target_error
from src.visual_servoing.approach_drive import compute_drive_command, MIN_SPEED, MAX_SPEED


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
    assert error.distance_cm is not None and error.distance_cm > 80, error
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
    """Centered and within STOP_DISTANCE_CM -> reached, ready for arm handoff."""
    result = _make_detection(norm_x=0.5, bbox_height_px=400)
    error = compute_target_error(result)
    assert error.found
    assert abs(error.lateral_error) < 1e-9, error
    assert error.distance_cm is not None and error.distance_cm <= 15, error
    assert error.reached
    print("PASS reached: centered + close")


# ── approach_drive.compute_drive_command ─────────────────────────────────

def test_drive_command_none_when_not_found():
    error = compute_target_error(DetectionResult())
    assert compute_drive_command(error, _kin()) is None
    print("PASS drive command is None when nothing is found")


def test_drive_command_none_when_reached():
    error = compute_target_error(_make_detection(norm_x=0.5, bbox_height_px=400))
    assert error.reached
    assert compute_drive_command(error, _kin()) is None
    print("PASS drive command is None when reached (arm handoff)")


def test_drive_left_when_can_is_left_of_center():
    """Far + left of center -> steers toward arc_forward_left (left wheel
    slower than right), speed ramped up near MAX_SPEED since it's far."""
    error = compute_target_error(_make_detection(norm_x=0.3, bbox_height_px=40))
    cmd = compute_drive_command(error, _kin())
    assert cmd is not None
    assert cmd.left_speed < cmd.right_speed, cmd
    assert cmd.left_speed > MIN_SPEED * 0.5, cmd   # brisk, not crawling
    print("PASS steers left + brisk speed when can is far left of center")


def test_drive_right_when_can_is_right_of_center():
    """Close + right of center -> steers toward arc_forward_right (right
    wheel slower than left), speed near MIN_SPEED since it's close."""
    error = compute_target_error(_make_detection(norm_x=0.7, bbox_height_px=200))
    cmd = compute_drive_command(error, _kin())
    assert cmd is not None
    assert cmd.right_speed < cmd.left_speed, cmd
    assert cmd.left_speed <= MAX_SPEED, cmd
    print("PASS steers right + gentler speed when can is close and right of center")


# ── Plain runner (no pytest needed) ───────────────────────────────────────

ALL_TESTS = [
    test_no_detection_not_found,
    test_far_left_of_center,
    test_close_right_of_center,
    test_reached_when_centered_and_close,
    test_drive_command_none_when_not_found,
    test_drive_command_none_when_reached,
    test_drive_left_when_can_is_left_of_center,
    test_drive_right_when_can_is_right_of_center,
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

def run_live_demo():
    """Poll the real camera, drive the real motors. Run by hand on the Pi:
        python tests/test_ibvs_centering.py --live
    Ctrl+C to stop."""
    import time
    from src.hardware.sensors.camera_sensor import CameraSensor
    from src.hardware.actuators.pwm_driver import PWMActuator

    cal = MotionCalibration()
    kin = DifferentialKinematics(cal)
    actuator = PWMActuator(calibration=cal)
    camera = CameraSensor()
    camera.start()
    try:
        while True:
            result = camera.get_latest_result()
            error = compute_target_error(result)
            cmd = compute_drive_command(error, kin)
            if cmd is None:
                actuator.stop()
                if error.reached:
                    print("[live] reached — ready for arm handoff")
            else:
                actuator.apply(cmd)
            print(f"[live] found={error.found} lateral={error.lateral_error:+.2f} "
                  f"dist_cm={error.distance_cm} reached={error.reached}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        actuator.stop()
        actuator.close()
        camera.stop()


if __name__ == "__main__":
    if "--live" in sys.argv:
        run_live_demo()
    else:
        _run_all_tests()
