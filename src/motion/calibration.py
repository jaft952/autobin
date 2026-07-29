from dataclasses import dataclass


@dataclass(frozen=True)
class MotorPins:
    # ZK-BM1 dual H-bridge: 4 input pins only, no ENA/ENB enable pins.
    # Speed is set by PWM-ing the direction inputs directly (see PWMActuator).
    #   Left  motor (OUT1/OUT2): in1 / in2
    #   Right motor (OUT3/OUT4): in3 / in4
    # AS WIRED 2026-07-13 (user). Bonus: BCM 12/13/18/19 are exactly the
    # Pi's four hardware-PWM-capable pins — a later switch to jitter-free
    # hardware PWM (pigpio etc.) would need no rewiring.
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

    motor_a_forward_trim: float = 1.0
    motor_b_forward_trim: float = 0.87
    motor_a_backward_trim: float = 0.7
    motor_b_backward_trim: float = 1.0
    motor_a_turn_trim: float = 1.0
    motor_b_turn_trim: float = 1.0


    # Direction flags — derived from the tests/test_wheels.py session on
    # 2026-07-13 with the ZK-BM1 wiring (IN1/2=left, IN3/4=right):
    #   observed IN1=left-forward (left motor polarity already correct) and
    #   IN4=right-forward (right motor mirrored, as differential drives are).
    # The old True/True/True values were for the L298N-era wiring — obsolete.
    invert_left: bool = False
    invert_right: bool = True

    swap_left_right: bool = False

    arc_inner_wheel_ratio: float = 0.3
