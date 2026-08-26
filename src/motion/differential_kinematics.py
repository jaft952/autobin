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
        """Forward-left arc. angle_deg: 0=straight, 90=sharp left turn, 180=spin left.
        speed: overrides the calibrated arc_speed baseline for this call, e.g. a
        dynamically computed speed from a proportional visual-servoing controller."""
        ratio = _ratio_from_angle(angle_deg) if angle_deg is not None else self.cal.arc_inner_wheel_ratio
        base = speed if speed is not None else self.cal.arc_speed
        return WheelCommand(base * ratio, base, True, "forward")

    def arc_forward_right(self, angle_deg: float | None = None, speed: float | None = None) -> WheelCommand:
        """Forward-right arc. angle_deg: 0=straight, 90=sharp right turn, 180=spin right.
        speed: overrides the calibrated arc_speed baseline for this call, e.g. a
        dynamically computed speed from a proportional visual-servoing controller."""
        ratio = _ratio_from_angle(angle_deg) if angle_deg is not None else self.cal.arc_inner_wheel_ratio
        base = speed if speed is not None else self.cal.arc_speed
        return WheelCommand(base, base * ratio, True, "forward")

    def arc_backward_left(self, angle_deg: float | None = None, speed: float | None = None) -> WheelCommand:
        """Backward-left arc. angle_deg: 0=straight, 90=sharp left turn, 180=spin left.
        speed: overrides the calibrated arc_speed baseline for this call."""
        ratio = _ratio_from_angle(angle_deg) if angle_deg is not None else self.cal.arc_inner_wheel_ratio
        base = speed if speed is not None else self.cal.arc_speed
        return WheelCommand(-base * ratio, -base, True, "backward")

    def arc_backward_right(self, angle_deg: float | None = None, speed: float | None = None) -> WheelCommand:
        """Backward-right arc. angle_deg: 0=straight, 90=sharp right turn, 180=spin right.
        speed: overrides the calibrated arc_speed baseline for this call."""
        ratio = _ratio_from_angle(angle_deg) if angle_deg is not None else self.cal.arc_inner_wheel_ratio
        base = speed if speed is not None else self.cal.arc_speed
        return WheelCommand(-base, -base * ratio, True, "backward")
