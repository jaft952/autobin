from dataclasses import dataclass
import math

from src.motion.calibration import MotionCalibration


@dataclass
class WheelCommand:
    left_speed: float
    right_speed: float
    apply_trim: bool = True
    trim_set: str = "forward"


def _ratio_from_angle(angle_deg: float) -> float:
    """Convert arc angle (degrees) to inner wheel ratio.

    angle_deg: 0 = straight (ratio 1.0), 90 = sharp turn (ratio ~0.33),
               180 = reverse turn (ratio 0.0 or negative)
    """
    if angle_deg == 0:
        return 1.0
    if angle_deg == 90:
        return 0.33  # roughly 90-degree arc
    if angle_deg == 180:
        return 0.0   # spin in place
    # Linear interpolation for intermediate values
    if 0 < angle_deg < 90:
        return 1.0 - (angle_deg / 90.0) * 0.67
    elif 90 < angle_deg < 180:
        return 0.33 - ((angle_deg - 90.0) / 90.0) * 0.33
    else:
        return 1.0  # default for out-of-range


class DifferentialKinematics:
    """Developer-facing interface for movement logic.

    This module is intentionally hardware-agnostic: it outputs wheel commands only.
    """

    def __init__(self, calibration: MotionCalibration | None = None) -> None:
        self.cal = calibration or MotionCalibration()

    def forward(self) -> WheelCommand:
        return WheelCommand(self.cal.forward_speed, self.cal.forward_speed, True, "forward")

    def backward(self) -> WheelCommand:
        return WheelCommand(-self.cal.backward_speed, -self.cal.backward_speed, True, "backward")

    def turn_left(self) -> WheelCommand:
        return WheelCommand(-self.cal.turn_speed, self.cal.turn_speed, True, "turn")

    def turn_right(self) -> WheelCommand:
        return WheelCommand(self.cal.turn_speed, -self.cal.turn_speed, True, "turn")

    def arc_forward_left(self, angle_deg: float | None = None) -> WheelCommand:
        """Forward-left arc. angle_deg: 0=straight, 90=sharp left turn, 180=spin left."""
        ratio = _ratio_from_angle(angle_deg) if angle_deg is not None else self.cal.arc_inner_wheel_ratio
        left = self.cal.arc_speed * ratio * self.cal.motor_a_forward_trim
        right = self.cal.arc_speed * self.cal.motor_b_forward_trim
        return WheelCommand(left, right, False, "forward")

    def arc_forward_right(self, angle_deg: float | None = None) -> WheelCommand:
        """Forward-right arc. angle_deg: 0=straight, 90=sharp right turn, 180=spin right."""
        ratio = _ratio_from_angle(angle_deg) if angle_deg is not None else self.cal.arc_inner_wheel_ratio
        left = self.cal.arc_speed * self.cal.motor_a_forward_trim
        right = self.cal.arc_speed * ratio * self.cal.motor_b_forward_trim
        return WheelCommand(left, right, False, "forward")

    def arc_backward_left(self, angle_deg: float | None = None) -> WheelCommand:
        """Backward-left arc. angle_deg: 0=straight, 90=sharp left turn, 180=spin left."""
        ratio = _ratio_from_angle(angle_deg) if angle_deg is not None else self.cal.arc_inner_wheel_ratio
        left = -self.cal.arc_speed * ratio * self.cal.motor_a_backward_trim
        right = -self.cal.arc_speed * self.cal.motor_b_backward_trim
        return WheelCommand(left, right, False, "backward")

    def arc_backward_right(self, angle_deg: float | None = None) -> WheelCommand:
        """Backward-right arc. angle_deg: 0=straight, 90=sharp right turn, 180=spin right."""
        ratio = _ratio_from_angle(angle_deg) if angle_deg is not None else self.cal.arc_inner_wheel_ratio
        left = -self.cal.arc_speed * self.cal.motor_a_backward_trim
        right = -self.cal.arc_speed * ratio * self.cal.motor_b_backward_trim
        return WheelCommand(left, right, False, "backward")
