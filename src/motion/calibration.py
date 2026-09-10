from dataclasses import dataclass


@dataclass(frozen=True)
class MotorPins:
    # ZK-BM1 dual H-bridge: 4 input pins, PWM'd directly, no ENA/ENB.
    in1: int = 12   # physical pin 32 -> ZK-BM1 IN1
    in2: int = 13   # physical pin 33 -> ZK-BM1 IN2
    in3: int = 18   # physical pin 12 -> ZK-BM1 IN3
    in4: int = 19   # physical pin 35 -> ZK-BM1 IN4


@dataclass(frozen=True)
class MotionCalibration:

    forward_speed: float = 90.0
    backward_speed: float = 90.0
    turn_speed: float = 90.0
    arc_speed: float = 90.0

    motor_a_forward_trim: float = 1.0 # right
    motor_b_forward_trim: float = 0.78 # left
    motor_a_backward_trim: float = 1.0 # right
    motor_b_backward_trim: float = 0.95 # left
    motor_a_turn_trim: float = 1.0
    motor_b_turn_trim: float = 1.0


    # Direction flags calibrated for the ZK-BM1 wiring.
    invert_left: bool = False
    invert_right: bool = True

    # Left/right channels are swapped in wiring; pivot direction was reversed without this.
    swap_left_right: bool = True

    arc_inner_wheel_ratio: float = 0.3
