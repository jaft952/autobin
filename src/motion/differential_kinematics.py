from dataclasses import dataclass

from src.motion.calibration import MotionCalibration


@dataclass(frozen=True)
class WheelCommand:
    left_speed: float
    right_speed: float
    apply_trim: bool = True
    trim_set: str = "forward"


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

    def arc_forward_left(self) -> WheelCommand:
        left = self.cal.arc_speed * self.cal.arc_inner_wheel_ratio * self.cal.motor_a_forward_trim
        right = self.cal.arc_speed * self.cal.motor_b_forward_trim
        return WheelCommand(left, right, False, "forward")

    def arc_forward_right(self) -> WheelCommand:
        left = self.cal.arc_speed * self.cal.motor_a_forward_trim
        right = self.cal.arc_speed * self.cal.arc_inner_wheel_ratio * self.cal.motor_b_forward_trim
        return WheelCommand(left, right, False, "forward")

    def arc_backward_left(self) -> WheelCommand:
        left = -self.cal.arc_speed * self.cal.arc_inner_wheel_ratio * self.cal.motor_a_backward_trim
        right = -self.cal.arc_speed * self.cal.motor_b_backward_trim
        return WheelCommand(left, right, False, "backward")

    def arc_backward_right(self) -> WheelCommand:
        left = -self.cal.arc_speed * self.cal.motor_a_backward_trim
        right = -self.cal.arc_speed * self.cal.arc_inner_wheel_ratio * self.cal.motor_b_backward_trim
        return WheelCommand(left, right, False, "backward")
