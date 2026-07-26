"""
src/visual_servoing/prediction_model.py

Vision measurement -> local Cartesian pose, and the physics of one predicted
control step. Pure functions/dataclasses, same convention as distance_error.py:
no hardware access, nothing mutated between calls.

Frame convention: this robot has no odometry or IMU, so there is no
persistent global pose to build on. Instead, every planning cycle defines its
OWN local frame: the robot is placed at the origin, heading along local +X,
theta measured counter-clockwise from +X. The target (can) is treated as
static within that one planning cycle -- it is not tracked across cycles,
only re-measured from the next camera frame.

Bearing sign is NOT derived from "which side is physically the robot's left"
-- reactive_controller.py documents a hardware-verified quirk where a positive
lateral_error (can right of frame center) is correctly corrected by calling
arc_forward_LEFT, not arc_forward_right, which is the opposite of what a
naive physical-left/right reading would suggest (see that module's docstring
for the on-hardware finding this came from). Re-deriving "true" physical left
from calibration alone would risk silently disagreeing with that empirical
result. Instead, +Y here is DEFINED as whichever side arc_forward_left's own
WheelCommand kinematics (positive omega, see step() below) turns the heading
toward, and bearing_from_lateral_error is defined to place a positive
lateral_error's target on that same +Y side. That makes the two facts agree
by construction: the action reactive_controller.py already validated as correct
on hardware is, in this simulator, also the one that reduces predicted error.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

from src.visual_servoing.distance_error import TargetError
from src.motion.differential_kinematics import WheelCommand
from src.motion.calibration import MotionCalibration

# Logitech Brio horizontal field of view. UNCALIBRATED placeholder -- measure
# on hardware: place the can at a known lateral offset and a known distance,
# compare the predicted vs actual normalized x (same measurement spirit as
# CALIBRATION_CONSTANT_PX_CM in distance_error.py).
CAMERA_HFOV_DEG = 70.0  # TODO calibrate on hardware

CONTROL_DT_S = 0.2  # TODO tune: assumed duration of one control step/tick


@dataclass
class RobotState:
    x_cm: float
    y_cm: float
    theta_rad: float


@dataclass
class TargetPoint:
    x_cm: float
    y_cm: float


@dataclass
class ActionStep:
    """One candidate (steer, speed) command. steer_deg uses the SAME sign
    convention as TargetError.lateral_error: positive = correcting toward a
    can that is right of frame center. See trajectory_planner.action_to_command
    for how that maps onto DifferentialKinematics.arc_forward_left/right --
    that mapping is hardware-verified (see reactive_controller.py) and must not be
    re-derived from first principles."""
    steer_deg: float
    speed: float


def bearing_from_lateral_error(lateral_error: float, hfov_deg: float = CAMERA_HFOV_DEG) -> float:
    """Normalized image-plane x offset (-0.5..0.5) -> bearing angle in
    radians, via the pinhole relation tan(bearing)/tan(half_fov) = 2*offset.
    Sign is NOT negated (see the module docstring's "Bearing sign" note): a
    positive lateral_error places the target at a positive bearing, which is
    the same side arc_forward_left's raw WheelCommand kinematics turn the
    heading toward -- matching the hardware-verified fact that arc_forward_left
    is the correct real-world action for a positive lateral_error.
    """
    half_fov = math.radians(hfov_deg) / 2.0
    return math.atan(2.0 * lateral_error * math.tan(half_fov))


def lateral_error_from_bearing(phi_rad: float, hfov_deg: float = CAMERA_HFOV_DEG) -> float:
    """Inverse of bearing_from_lateral_error. Clipped to +-0.5 (frame edges)
    -- a predicted bearing beyond the camera's half-FOV means the target is
    predicted to have left the frame entirely."""
    half_fov = math.radians(hfov_deg) / 2.0
    e = math.tan(phi_rad) / (2.0 * math.tan(half_fov))
    return max(-0.5, min(0.5, e))


def estimate_initial_state(error: TargetError,
                            hfov_deg: float = CAMERA_HFOV_DEG) -> Tuple[RobotState, TargetPoint]:
    """Build the local frame for one planning cycle: robot at the origin,
    target placed at the polar position implied by this frame's TargetError.

    Precondition (caller's responsibility, same as reactive_controller.py's use of
    TargetError): error.found is True and error.distance_cm is not None --
    the not-found/too-close/reached cases are short-circuited before planning
    ever runs, so this never has to handle them.
    """
    robot = RobotState(x_cm=0.0, y_cm=0.0, theta_rad=0.0)
    r = error.distance_cm
    phi = bearing_from_lateral_error(error.lateral_error, hfov_deg)
    target = TargetPoint(x_cm=r * math.cos(phi), y_cm=r * math.sin(phi)) # type: ignore
    return robot, target


def step(state: RobotState, cmd: WheelCommand, dt_s: float,
         cal: MotionCalibration) -> RobotState:
    """Advance one control step via differential-drive (unicycle) kinematics:

        v_L = k * left_speed,  v_R = k * right_speed      (cm/s)
        v = (v_L + v_R) / 2                                 (forward speed)
        omega = (v_R - v_L) / track_width_cm                (rad/s, CCW+)
        x' = x + v*cos(theta)*dt
        y' = y + v*sin(theta)*dt
        theta' = theta + omega*dt

    cmd.left_speed/right_speed are clamped to +-100 first, mirroring the
    clamp PWMActuator.apply() applies before the values ever reach the motors
    -- the simulator should never assume a physical speed the hardware
    couldn't actually deliver.

    Deliberately does NOT replicate PWMActuator's invert_left/invert_right/
    swap_left_right wiring correction -- that correction (plus whatever else
    is behind the arc-steering quirk reactive_controller.py found on hardware) is
    already absorbed on the OTHER end of this model, in
    bearing_from_lateral_error's sign convention (see the module docstring).
    Applying it here too would double-correct.
    """
    left = max(-100.0, min(100.0, cmd.left_speed))
    right = max(-100.0, min(100.0, cmd.right_speed))

    k = cal.wheel_speed_cm_per_s_per_unit
    v_l = k * left
    v_r = k * right

    v = (v_l + v_r) / 2.0
    omega = (v_r - v_l) / cal.track_width_cm

    return RobotState(
        x_cm=state.x_cm + v * math.cos(state.theta_rad) * dt_s,
        y_cm=state.y_cm + v * math.sin(state.theta_rad) * dt_s,
        theta_rad=state.theta_rad + omega * dt_s,
    )


def predicted_error(state: RobotState, target: TargetPoint,
                     hfov_deg: float = CAMERA_HFOV_DEG) -> Tuple[float, float]:
    """-> (predicted_lateral_error, predicted_distance_cm): the static
    target's position reprojected into `state`, through the same pinhole
    geometry used to interpret the original vision measurement -- so the
    output lands in exactly the units trajectory_planner.py (and
    distance_error.py's STOP_DISTANCE_CM/CENTER_TOLERANCE) already reason about.
    """
    dx = target.x_cm - state.x_cm
    dy = target.y_cm - state.y_cm
    distance = math.hypot(dx, dy)
    absolute_bearing = math.atan2(dy, dx)
    phi = _wrap_to_pi(absolute_bearing - state.theta_rad)
    lateral_error = lateral_error_from_bearing(phi, hfov_deg)
    return lateral_error, distance


def _wrap_to_pi(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2 * math.pi) - math.pi
