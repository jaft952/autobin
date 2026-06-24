from dataclasses import dataclass


@dataclass(frozen=True)
class MotorPins:
    in1: int = 17
    in2: int = 27
    in3: int = 22
    in4: int = 23
    ena: int = 18
    enb: int = 19


@dataclass(frozen=True)
class MotionCalibration:
    forward_speed: float = 65.0
    backward_speed: float = 65.0
    turn_speed: float = 60.0
    arc_speed: float = 65.0

    motor_a_forward_trim: float = 1.0
    motor_b_forward_trim: float = 1.0
    motor_a_backward_trim: float = 1.0
    motor_b_backward_trim: float = 1.0
    motor_a_turn_trim: float = 1.0
    motor_b_turn_trim: float = 1.0

    arc_inner_wheel_ratio: float = 0.3
