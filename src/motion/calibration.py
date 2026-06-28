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
    backward_speed: float = 50.0
    turn_speed: float = 40.0
    arc_speed: float = 50.0

    motor_a_forward_trim: float = 1.0
    motor_b_forward_trim: float = 1.0
    motor_a_backward_trim: float = 1.0
    motor_b_backward_trim: float = 1.0
    motor_a_turn_trim: float = 1.0
    motor_b_turn_trim: float = 1.0

    # Motor wiring polarity. The wheels turned OPPOSITE to the commanded move
    # (FORWARD drove backward), so both motors are wired reversed -> flip each
    # one in software.
    invert_left: bool = True
    invert_right: bool = True

    # The two drive motors are plugged into each other's channels (A on the right,
    # B on the left), so FORWARD looked fine but LEFT/RIGHT turns came out reversed.
    # Swap the left/right wheel speeds in software to undo it.
    swap_left_right: bool = True

    arc_inner_wheel_ratio: float = 0.3
