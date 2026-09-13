from dataclasses import dataclass


@dataclass(frozen=True)
class MotorPins:
    in1: int = 12
    in2: int = 13
    in3: int = 18
    in4: int = 19


@dataclass(frozen=True)
class MotionCalibration:

    forward_speed: float = 90.0
    backward_speed: float = 90.0
    turn_speed: float = 90.0
    arc_speed: float = 90.0

    motor_a_forward_trim: float = 1.0
    motor_b_forward_trim: float = 0.75
    motor_a_backward_trim: float = 1.0
    motor_b_backward_trim: float = 0.95
    motor_a_turn_trim: float = 1.0
    motor_b_turn_trim: float = 1.0

    invert_left: bool = False
    invert_right: bool = True

    swap_left_right: bool = True

    arc_inner_wheel_ratio: float = 0.3
