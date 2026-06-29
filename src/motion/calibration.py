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

    forward_speed: float = 70.0
    backward_speed: float = 70.0
    turn_speed: float = 70.0
    arc_speed: float = 70.0

    motor_a_forward_trim: float = 1.0
    motor_b_forward_trim: float = 1.0
    motor_a_backward_trim: float = 1.0
    motor_b_backward_trim: float = 1.0
    motor_a_turn_trim: float = 1.0
    motor_b_turn_trim: float = 1.0


    invert_left: bool = True
    invert_right: bool = True


    swap_left_right: bool = True

    arc_inner_wheel_ratio: float = 0.3
