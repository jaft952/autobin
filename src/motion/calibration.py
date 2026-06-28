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
    # Duty-cycle % per move. Kept high enough to overcome motor stall, but slower
    # than before so each pulsed nudge is gentle. turn_speed is the lowest because
    # turning overshoots the most. If the base won't move at all, raise these; if
    # it still overshoots, shorten DRIVE_PULSE_S in chassis_controller.py instead.
    forward_speed: float = 40.0
    backward_speed: float = 40.0
    turn_speed: float = 40.0
    arc_speed: float = 50.0

    motor_a_forward_trim: float = 1.0
    motor_b_forward_trim: float = 1.0
    motor_a_backward_trim: float = 1.0
    motor_b_backward_trim: float = 1.0
    motor_a_turn_trim: float = 1.0
    motor_b_turn_trim: float = 1.0

    arc_inner_wheel_ratio: float = 0.3
