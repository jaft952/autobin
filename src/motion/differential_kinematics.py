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
    """Convert an arc angle into the inner wheel speed ratio."""
    if angle_deg == 0:
        return 1.0
    if angle_deg == 90:
        return 0.33
    if angle_deg == 180:
        return 0.0
    if 0 < angle_deg < 90:
        return 1.0 - (angle_deg / 90.0) * 0.67
    elif 90 < angle_deg < 180:
        return 0.33 - ((angle_deg - 90.0) / 90.0) * 0.33
    else:
        return 1.0


class DifferentialKinematics:
    """Turns movement requests into left and right wheel speeds."""

    def __init__(self, calibration: MotionCalibration | None = None) -> None:
        self.cal = calibration or MotionCalibration()

    def forward(self, speed: float | None = None) -> WheelCommand:
        s = speed if speed is not None else self.cal.forward_speed
        return WheelCommand(s, s, True, "forward")

    def backward(self, speed: float | None = None) -> WheelCommand:
        s = speed if speed is not None else self.cal.backward_speed
        return WheelCommand(-s, -s, True, "backward")

    def turn_left(self, speed: float | None = None) -> WheelCommand:
        s = speed if speed is not None else self.cal.turn_speed
        return WheelCommand(-s, s, True, "turn")

    def turn_right(self, speed: float | None = None) -> WheelCommand:
        s = speed if speed is not None else self.cal.turn_speed
        return WheelCommand(s, -s, True, "turn")

    def arc_forward_left(self, angle_deg: float | None = None, speed: float | None = None) -> WheelCommand:
        """Drive forward in a left arc."""
        ratio = _ratio_from_angle(angle_deg) if angle_deg is not None else self.cal.arc_inner_wheel_ratio
        base = speed if speed is not None else self.cal.arc_speed
        return WheelCommand(base * ratio, base, True, "forward")

    def arc_forward_right(self, angle_deg: float | None = None, speed: float | None = None) -> WheelCommand:
        """Drive forward in a right arc."""
        ratio = _ratio_from_angle(angle_deg) if angle_deg is not None else self.cal.arc_inner_wheel_ratio
        base = speed if speed is not None else self.cal.arc_speed
        return WheelCommand(base, base * ratio, True, "forward")

    def arc_backward_left(self, angle_deg: float | None = None, speed: float | None = None) -> WheelCommand:
        """Drive backward in a left arc."""
        ratio = _ratio_from_angle(angle_deg) if angle_deg is not None else self.cal.arc_inner_wheel_ratio
        base = speed if speed is not None else self.cal.arc_speed
        return WheelCommand(-base * ratio, -base, True, "backward")

    def arc_backward_right(self, angle_deg: float | None = None, speed: float | None = None) -> WheelCommand:
        """Drive backward in a right arc."""
        ratio = _ratio_from_angle(angle_deg) if angle_deg is not None else self.cal.arc_inner_wheel_ratio
        base = speed if speed is not None else self.cal.arc_speed
        return WheelCommand(-base, -base * ratio, True, "backward")
