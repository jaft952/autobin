"""
tests/test_predictive_controller.py

Pure-logic tests for prediction_model.py + cost_function.py +
trajectory_planner.py + predictive_controller.py. NO hardware or camera
needed — same spirit as tests/test_ibvs_centering.py: synthetic
DetectionResult/BoundingBox fixtures, exact/behavioral assertions.

    python tests/test_predictive_controller.py      # plain runner with PASS/FAIL summary
    pytest tests/test_predictive_controller.py       # also works

For a live run on the robot, use tests/test_ibvs_centering.py's --live
--drive --predictive flags instead — this file only exercises pure functions.
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.perception.detector import BoundingBox, DetectionResult
from src.motion.calibration import MotionCalibration
from src.motion.differential_kinematics import DifferentialKinematics, WheelCommand
from src.visual_servoing.distance_error import compute_target_error, TargetError, STOP_DISTANCE_CM
from src.visual_servoing.prediction_model import (
    ActionStep, RobotState, TargetPoint,
    bearing_from_lateral_error, lateral_error_from_bearing, step as physics_step,
)
from src.visual_servoing.cost_function import evaluate, MIN_SAFE_DISTANCE_CM
from src.visual_servoing.trajectory_planner import plan, action_to_command
from src.visual_servoing.predictive_controller import ControllerState, compute_predictive_command


# ── Fixture builder (same as test_ibvs_centering.py) ────────────────────

def _make_detection(norm_x: float, bbox_height_px: float,
                     frame_width: int = 640, frame_height: int = 480) -> DetectionResult:
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


def _target_error(lateral_error: float, distance_cm: float,
                   reached: bool = False, too_close: bool = False) -> TargetError:
    """Build a TargetError directly rather than via _make_detection() for the
    planning-logic tests below -- they need a realistic, moderate
    distance_cm to exercise steering choice, and routing that through
    distance_error.py's pixel-height calibration would couple these tests to
    CALIBRATION_CONSTANT_PX_CM (an UNCALIBRATED, hardware-measured placeholder
    that changes independently of this planner and, at its current value,
    cannot represent moderate distances with a bbox that fits in any
    reasonable frame height at all -- see distance_error.py's own note on it).
    Steering direction and the planner's cost tradeoffs don't depend on how
    distance_cm was derived, only on its value, so this is a more direct unit
    of test for trajectory_planner.py / predictive_controller.py anyway.
    """
    return TargetError(found=True, lateral_error=lateral_error, distance_cm=distance_cm,
                        reached=reached, too_close=too_close)


ZERO_ACTION = ActionStep(steer_deg=0.0, speed=0.0)


# ── prediction_model: pinhole geometry ───────────────────────────────────

def test_bearing_round_trip():
    for e in (-0.4, -0.2, -0.05, 0.0, 0.05, 0.2, 0.4):
        phi = bearing_from_lateral_error(e)
        back = lateral_error_from_bearing(phi)
        assert abs(back - e) < 1e-9, (e, phi, back)
    print("PASS bearing_from_lateral_error / lateral_error_from_bearing round-trip")


def test_bearing_sign_matches_lateral_error_sign():
    """Positive lateral_error (can right) -> positive bearing; this is a
    deliberate bookkeeping choice (see prediction_model.py's module
    docstring), not a claim about physical left/right, chosen so that
    arc_forward_left's own WheelCommand kinematics (positive omega) agree
    with the hardware-verified fact that arc_forward_left is the correct
    real-world action for a positive lateral_error (see approach_drive.py)."""
    assert bearing_from_lateral_error(0.3) > 0
    assert bearing_from_lateral_error(-0.3) < 0
    assert bearing_from_lateral_error(0.0) == 0.0
    print("PASS bearing sign matches lateral_error sign")


# ── prediction_model: kinematics ─────────────────────────────────────────

def test_step_straight_preserves_heading():
    """Equal left/right wheel speeds -> omega=0 -> theta and y unchanged,
    x increases."""
    cal = MotionCalibration()
    state = RobotState(x_cm=0.0, y_cm=0.0, theta_rad=0.0)
    cmd = WheelCommand(left_speed=20.0, right_speed=20.0)
    new_state = physics_step(state, cmd, dt_s=0.2, cal=cal)
    assert new_state.x_cm > 0, new_state
    assert abs(new_state.y_cm) < 1e-9, new_state
    assert abs(new_state.theta_rad) < 1e-9, new_state
    print("PASS straight WheelCommand preserves heading, moves forward only")


def test_step_turn_sign():
    """Right wheel faster -> positive omega (theta increases); left wheel
    faster -> negative omega (theta decreases) -- the plain textbook
    differential-drive formula, independent of any hardware wiring quirk
    (see prediction_model.step's docstring for why no invert correction is
    applied here)."""
    cal = MotionCalibration()
    state = RobotState(x_cm=0.0, y_cm=0.0, theta_rad=0.0)

    right_faster = physics_step(state, WheelCommand(left_speed=10.0, right_speed=20.0), 0.2, cal)
    assert right_faster.theta_rad > 0, right_faster

    left_faster = physics_step(state, WheelCommand(left_speed=20.0, right_speed=10.0), 0.2, cal)
    assert left_faster.theta_rad < 0, left_faster
    print("PASS turn sign: right-wheel-faster turns +theta, left-wheel-faster turns -theta")


# ── trajectory_planner: action_to_command hardware mapping ──────────────

def test_action_to_command_matches_hardware_verified_mapping():
    """Copied convention from approach_drive.py: a positive steer_deg (used
    to correct a can right of center) must produce right_speed > left_speed
    (arc_forward_left), and negative steer_deg must produce
    left_speed > right_speed (arc_forward_right) -- matching
    approach_drive.py's own test_drive_when_can_is_right_of_center /
    test_drive_when_can_is_left_of_center assertions exactly."""
    kin = _kin()
    cmd_pos = action_to_command(ActionStep(steer_deg=20.0, speed=30.0), kin)
    assert cmd_pos.left_speed < cmd_pos.right_speed, cmd_pos

    cmd_neg = action_to_command(ActionStep(steer_deg=-20.0, speed=30.0), kin)
    assert cmd_neg.right_speed < cmd_neg.left_speed, cmd_neg
    print("PASS action_to_command steer sign matches approach_drive's hardware-verified mapping")


# ── trajectory_planner: end-to-end planning ──────────────────────────────

def test_plan_steers_toward_can_right_of_center():
    error = _target_error(lateral_error=0.2, distance_cm=60.0)
    result = plan(error, _kin(), prev_action=ZERO_ACTION)
    assert result.actions[0].steer_deg > 0, result.actions[0]
    print("PASS plan picks positive steer (arc_forward_left) for can right of center")


def test_plan_steers_toward_can_left_of_center():
    error = _target_error(lateral_error=-0.2, distance_cm=60.0)
    result = plan(error, _kin(), prev_action=ZERO_ACTION)
    assert result.actions[0].steer_deg < 0, result.actions[0]
    print("PASS plan picks negative steer (arc_forward_right) for can left of center")


def test_plan_is_deterministic():
    error = _target_error(lateral_error=0.15, distance_cm=45.0)
    kin = _kin()
    first = plan(error, kin, prev_action=ZERO_ACTION)
    second = plan(error, kin, prev_action=ZERO_ACTION)
    assert [a.steer_deg for a in first.actions] == [a.steer_deg for a in second.actions]
    assert [a.speed for a in first.actions] == [a.speed for a in second.actions]
    assert first.cost == second.cost
    print("PASS plan() is deterministic for identical inputs (beam search, no RNG)")


# ── cost_function ─────────────────────────────────────────────────────────

def test_cost_zero_at_stop_distance_centered_no_oscillation():
    action = ActionStep(steer_deg=0.0, speed=0.0)
    cost = evaluate([action], [0.0], [STOP_DISTANCE_CM], prev_action=action)
    assert abs(cost) < 1e-9, cost
    print("PASS cost is exactly zero when centered, at STOP_DISTANCE_CM, no oscillation")


def test_cost_penalizes_steer_oscillation():
    predicted_lateral = [0.1, 0.1, 0.1]
    predicted_distance = [20.0, 20.0, 20.0]
    prev = ActionStep(steer_deg=10.0, speed=20.0)

    smooth = [ActionStep(10.0, 20.0)] * 3
    oscillating = [ActionStep(10.0, 20.0), ActionStep(-10.0, 20.0), ActionStep(10.0, 20.0)]

    cost_smooth = evaluate(smooth, predicted_lateral, predicted_distance, prev)
    cost_osc = evaluate(oscillating, predicted_lateral, predicted_distance, prev)
    assert cost_osc > cost_smooth, (cost_smooth, cost_osc)
    print("PASS oscillating steer sequence costs more than a smooth one")


def test_cost_penalizes_overshoot():
    action = ActionStep(steer_deg=0.0, speed=0.0)
    safe = evaluate([action], [0.0], [MIN_SAFE_DISTANCE_CM + 5.0], prev_action=action)
    overshoot = evaluate([action], [0.0], [MIN_SAFE_DISTANCE_CM - 5.0], prev_action=action)
    assert overshoot > safe, (safe, overshoot)
    print("PASS predicted distance inside MIN_SAFE_DISTANCE_CM costs more (overshoot hinge)")


# ── predictive_controller: short-circuits (mirrors approach_drive tests) ─

def test_predictive_none_when_not_found():
    error = compute_target_error(DetectionResult())
    cmd, _ = compute_predictive_command(error, _kin(), ControllerState())
    assert cmd is None
    print("PASS predictive command is None when nothing is found")


def test_predictive_none_when_reached():
    error = _target_error(lateral_error=0.0, distance_cm=10.0, reached=True)
    cmd, _ = compute_predictive_command(error, _kin(), ControllerState())
    assert cmd is None
    print("PASS predictive command is None when reached (arm handoff)")


def test_predictive_backs_away_when_too_close():
    error = compute_target_error(_make_detection(norm_x=0.3, bbox_height_px=360))
    assert error.too_close
    cmd, _ = compute_predictive_command(error, _kin(), ControllerState())
    assert cmd is not None
    assert cmd.left_speed < 0 and cmd.right_speed < 0, cmd
    print("PASS predictive controller backs away (both wheels reverse) when too_close")


def test_predictive_steers_toward_can_right_of_center():
    error = _target_error(lateral_error=0.2, distance_cm=60.0)
    cmd, _ = compute_predictive_command(error, _kin(), ControllerState())
    assert cmd is not None
    assert cmd.left_speed < cmd.right_speed, cmd
    print("PASS predictive controller steers toward can right of center")


def test_predictive_steers_toward_can_left_of_center():
    error = _target_error(lateral_error=-0.2, distance_cm=60.0)
    cmd, _ = compute_predictive_command(error, _kin(), ControllerState())
    assert cmd is not None
    assert cmd.right_speed < cmd.left_speed, cmd
    print("PASS predictive controller steers toward can left of center")


def test_predictive_command_is_deterministic():
    error = compute_target_error(_make_detection(norm_x=0.35, bbox_height_px=60))
    kin = _kin()
    cmd1, state1 = compute_predictive_command(error, kin, ControllerState())
    cmd2, state2 = compute_predictive_command(error, kin, ControllerState())
    assert cmd1 is not None and cmd2 is not None
    assert cmd1.left_speed == cmd2.left_speed and cmd1.right_speed == cmd2.right_speed
    assert state1.prev_action.steer_deg == state2.prev_action.steer_deg
    print("PASS compute_predictive_command is deterministic for identical inputs")


# ── Plain runner (no pytest needed) ───────────────────────────────────────

ALL_TESTS = [
    test_bearing_round_trip,
    test_bearing_sign_matches_lateral_error_sign,
    test_step_straight_preserves_heading,
    test_step_turn_sign,
    test_action_to_command_matches_hardware_verified_mapping,
    test_plan_steers_toward_can_right_of_center,
    test_plan_steers_toward_can_left_of_center,
    test_plan_is_deterministic,
    test_cost_zero_at_stop_distance_centered_no_oscillation,
    test_cost_penalizes_steer_oscillation,
    test_cost_penalizes_overshoot,
    test_predictive_none_when_not_found,
    test_predictive_none_when_reached,
    test_predictive_backs_away_when_too_close,
    test_predictive_steers_toward_can_right_of_center,
    test_predictive_steers_toward_can_left_of_center,
    test_predictive_command_is_deterministic,
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
