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
code that touches GPIO — exactly mirroring how _ArmController wraps ArmActuator
for grasping in tests/test_ibvs_centering.py.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, Optional

from src.motion.calibration import MotionCalibration, MotorPins
from src.motion.differential_kinematics import DifferentialKinematics, WheelCommand
from src.visual_servoing.ibvs_centering import ChassisMove

# How long the base actually moves per vision tick when pulsing. The IBVS loop
# only updates every few (slow) YOLO frames; running the motors continuously
# between those updates overshoots. One short burst per tick = one small nudge.
DRIVE_PULSE_S = 0.12


class ChassisController:
    """Drives the differential base to satisfy an IBVS ChassisMove suggestion.

    One control tick = one call to apply(move). The motors run continuously at
    the calibrated speed for that move until the next apply()/stop(), so calling
    apply() once per IBVS update is enough to keep the base moving between the
    (slower) YOLO inference ticks.
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
            # Lazy import so merely importing this module never forces RPi.GPIO
            # to load. PWMActuator falls back to a MockGPIO off the Pi, so this
            # still constructs (and no-ops) on the dev PC.
            from src.hardware.actuators.pwm_driver import PWMActuator
            actuator = PWMActuator(pins=pins, calibration=self.cal)
        self.actuator = actuator

        # ChassisMove -> the DifferentialKinematics factory for its WheelCommand.
        # HOLD and SEARCH both map to None (stop): HOLD = centered, SEARCH = no
        # tin in view -> the base waits in place instead of spinning to scan.
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

    @property
    def last_move(self) -> Optional[ChassisMove]:
        return self._last

    def apply(self, move: ChassisMove) -> None:
        """Drive the base for one tick to satisfy `move`. HOLD (aligned) stops."""
        make_cmd = self._move_table.get(move)
        if make_cmd is None:
            self.stop()
        else:
            self.actuator.apply(make_cmd())
            self._last = move

    def pulse(self, move: ChassisMove, seconds: float = DRIVE_PULSE_S) -> None:
        """Move for one short burst, then stop — the pulsed form of apply().

        Use this (not apply) when the control loop ticks slowly: at each vision
        update the base nudges briefly and then holds still, so it can't overshoot
        the tin and spin while waiting for the next (slow) YOLO frame. HOLD and
        SEARCH have no motion, so this just stops for them.
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

    def close(self) -> None:
        """Stop and release GPIO. Call once on shutdown."""
        try:
            self.actuator.close()
        finally:
            self._last = None
