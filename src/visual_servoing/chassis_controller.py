"""
src/visual_servoing/chassis_controller.py

Chassis layer — turns IBVS centering suggestions into real wheel motion.

IBVSCentering produces a discrete ChassisMove each tick (FORWARD / TURN_LEFT /
FORWARD_LEFT / HOLD / SEARCH ...). This class is the bridge from that suggestion
to the differential-drive base:

    ChassisMove --DifferentialKinematics--> WheelCommand --PWMActuator--> motors

Keeping it here (not inside ibvs_centering.py) preserves the existing layering:
ibvs_centering stays hardware-free and unit-testable, src/motion stays a pure
hardware-agnostic package, and this orchestration module is the only chassis
code that touches GPIO.

--- MOTION STRATEGY ---
The old approach used pulse() with time.sleep() inside it, which blocked the
entire loop for up to 0.15s every tick on top of YOLO inference time (~0.5-1s
on Pi CPU). This caused the 1-second delay between movements.

The new approach uses apply() which runs motors CONTINUOUSLY between YOLO ticks.
YOLO inference time itself acts as the natural tick rate — the motor keeps moving
while YOLO processes the next frame, then direction is updated on the next result.
This gives smooth continuous motion instead of stop-start jerky movement.

Speed is now proportional to error magnitude via the speed scale passed to the
kinematics layer — big error = faster approach, near sweet spot = slow and gentle.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, Optional

from src.motion.calibration import MotionCalibration, MotorPins
from src.motion.differential_kinematics import DifferentialKinematics, WheelCommand
from src.visual_servoing.ibvs_centering import ChassisMove

# Speed scaling: error_mag (~0..1) is multiplied by SPEED_GAIN then clamped.
# Far from target -> high speed. Near target -> slow and gentle.
# Tune on the Pi:
#   SPEED_MIN too small -> motors stall near target        (raise it)
#   SPEED_MIN too big   -> overshoots near target          (lower it)
#   SPEED_MAX           -> approach speed when tin is far
#   SPEED_GAIN          -> how quickly speed grows with error
SPEED_MIN = 0.70         # minimum motor speed fraction (0..1) to prevent stall
SPEED_MAX = 1.0          # maximum motor speed fraction
SPEED_MAX_TURN = 0.8     # maximum turn speed fraction (0..1) to prevent overshoot
SPEED_GAIN = 3           # multiplier: error_mag * SPEED_GAIN = raw speed
SPEED_MAX_BACKWARD = 0.7 # limit backward speed to prevent overshoot

# Legacy fallback for plain pulse() calls (kept for backward compatibility)
DRIVE_PULSE_MIN = 0.05
DRIVE_PULSE_MAX = 0.15
DRIVE_PULSE_GAIN = 1.0
DRIVE_PULSE_S = 0.12


class ChassisController:
    """Drives the differential base to satisfy an IBVS ChassisMove suggestion.

    One control tick = one call to apply_for_error(move, error_mag).
    The motors run continuously at a speed proportional to the error magnitude
    until the next call, so the base keeps moving smoothly between the slower
    YOLO inference ticks instead of pulsing and stopping each tick.
    """

    def __init__(
        self,
        actuator=None,
        calibration: Optional[MotionCalibration] = None,
        pins: Optional[MotorPins] = None,
        kinematics: Optional[DifferentialKinematics] = None,
    ) -> None:
        self.cal = calibration or MotionCalibration()
        self.kin = kinematics or DifferentialKinematics(self.cal)
        if actuator is None:
            from src.hardware.actuators.pwm_driver import PWMActuator
            actuator = PWMActuator(pins=pins, calibration=self.cal)
        self.actuator = actuator

        self._move_table: Dict[ChassisMove, Optional[Callable[[], WheelCommand]]] = {
            ChassisMove.FORWARD: self.kin.forward,
            ChassisMove.BACKWARD: self.kin.backward,
            ChassisMove.TURN_LEFT: self.kin.turn_left,
            ChassisMove.TURN_RIGHT: self.kin.turn_right,
            ChassisMove.FORWARD_LEFT: self.kin.arc_forward_left,
            ChassisMove.FORWARD_RIGHT: self.kin.arc_forward_right,
            ChassisMove.BACKWARD_LEFT: self.kin.arc_backward_left,
            ChassisMove.BACKWARD_RIGHT: self.kin.arc_backward_right,
            ChassisMove.SEARCH: None,
            ChassisMove.HOLD: None,
        }
        self._last: Optional[ChassisMove] = None
        self._current_move: Optional[ChassisMove] = None

    @property
    def last_move(self) -> Optional[ChassisMove]:
        return self._last

    def apply(self, move: ChassisMove) -> None:
        """Drive the base continuously for this move. HOLD/SEARCH stops.

        Call once per YOLO tick — motors keep running at full calibrated speed
        until the next apply() or stop() call. Use apply_for_error() instead
        when you want proportional speed scaling.
        """
        make_cmd = self._move_table.get(move)
        if make_cmd is None:
            self.stop()
        else:
            self.actuator.apply(make_cmd())
            self._last = move
            self._current_move = move

    def apply_for_error(self, move: ChassisMove, error_mag: float) -> None:
        """Drive continuously with speed PROPORTIONAL to error_mag (~0..1).

        This is the main method to call each YOLO tick for smooth motion:
        - Big error (tin far from sweet spot) -> fast approach
        - Small error (tin near sweet spot)   -> slow and gentle
        - HOLD / SEARCH                        -> stop

        The motor keeps running between YOLO ticks naturally — no sleep needed.
        Direction updates automatically on the next tick when YOLO returns a
        new result.
        """
        make_cmd = self._move_table.get(move)
        if make_cmd is None:
            self.stop()
            return

        # Scale speed proportionally to error, clamped between min and max
        if move in (ChassisMove.BACKWARD, ChassisMove.BACKWARD_LEFT, ChassisMove.BACKWARD_RIGHT):
            speed_max = SPEED_MAX_BACKWARD
        elif move in (ChassisMove.TURN_LEFT, ChassisMove.TURN_RIGHT):
            speed_max = SPEED_MAX_TURN
        else:
            speed_max = SPEED_MAX
            
        speed = max(SPEED_MIN, min(speed_max, error_mag * SPEED_GAIN))

        cmd = make_cmd()

        # Scale the wheel command speeds if WheelCommand supports it
        # If your WheelCommand has left_speed / right_speed attributes, scale them
        if hasattr(cmd, "left_speed") and hasattr(cmd, "right_speed"):
            cmd.left_speed *= speed
            cmd.right_speed *= speed

        self.actuator.apply(cmd)
        self._last = move
        self._current_move = move

    def pulse_for_error(self, move: ChassisMove, error_mag: float) -> None:
        """Legacy pulsed method kept for backward compatibility.

        Prefer apply_for_error() for smooth continuous motion.
        This still uses sleep internally — use only if your loop explicitly
        needs the old pulse-and-stop behaviour.
        """
        seconds = max(DRIVE_PULSE_MIN, min(DRIVE_PULSE_MAX, error_mag * DRIVE_PULSE_GAIN))
        self.pulse(move, seconds)

    def pulse(self, move: ChassisMove, seconds: float = DRIVE_PULSE_S) -> None:
        """Legacy: move for one short burst then stop.

        Kept for backward compatibility. Prefer apply_for_error() for the
        live loop — this blocks for `seconds` which causes the jerky 1-second
        delay between movements when combined with YOLO inference time.
        """
        make_cmd = self._move_table.get(move)
        if make_cmd is None:
            self.stop()
            return
        self.actuator.apply(make_cmd())
        self._last = move
        time.sleep(seconds)
        self.stop()

    def stop(self) -> None:
        """Cut motor power and hold position (does not release GPIO)."""
        self.actuator.stop()
        self._last = ChassisMove.HOLD
        self._current_move = None

    def close(self) -> None:
        """Stop and release GPIO. Call once on shutdown."""
        try:
            self.actuator.close()
        finally:
            self._last = None
            self._current_move = None