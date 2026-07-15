from dataclasses import dataclass


@dataclass(frozen=True)
class MotorPins:
    # ZK-BM1 dual H-bridge: 4 input pins only, no ENA/ENB enable pins.
    # Speed is set by PWM-ing the direction inputs directly (see PWMActuator).
    #   Left  motor (A): in1 / in2
    #   Right motor (B): in3 / in4
    in1: int = 17
    in2: int = 27
    in3: int = 22
    in4: int = 23


@dataclass(frozen=True)
class MotionCalibration:

    forward_speed: float = 100.0
    backward_speed: float = 100.0
    turn_speed: float = 100.0
    arc_speed: float = 100.0

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
