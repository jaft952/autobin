"""
src/subsumption/motion_executor.py

MotionExecutor — turns the Arbitrator's winning ActionCommand into wheel
motion on the differential base.

Layers emit abstract motion_vector tuples and are PROHIBITED from touching
actuators (Rule 1); hardware access must go through src/hardware/actuators
(Rule 2). This module is the one place where, AFTER arbitration, the winning
vector becomes a WheelCommand for the PWMActuator:

    ActionCommand.motion_vector --mix--> WheelCommand --PWMActuator--> motors

motion_vector convention (v_x, v_y, v_theta):
    v_x      forward fraction  -1..1  (negative = reverse)
    v_y      ignored — the base is non-holonomic, it cannot strafe
    v_theta  turn fraction     -1..1, POSITIVE = CCW (left), matching
             DifferentialKinematics.turn_left() = (-speed, +speed)

Standard differential mix, renormalized so a hard turn while driving never
clips one wheel at 100% and silently straightens the arc:

    left  = v_x - v_theta
    right = v_x + v_theta
"""
from __future__ import annotations

from typing import Optional

from src.motion.calibration import MotionCalibration, MotorPins
from src.motion.differential_kinematics import WheelCommand
from src.subsumption.arbitrator import ActionCommand


class MotionExecutor:
    """Executes the winning ActionCommand on the wheels. One per robot."""

    def __init__(
        self,
        actuator=None,
        calibration: Optional[MotionCalibration] = None,
        pins: Optional[MotorPins] = None,
    ) -> None:
        self.cal = calibration or MotionCalibration()
        if actuator is None:
            from src.hardware.actuators.pwm_driver import PWMActuator
            actuator = PWMActuator(pins=pins, calibration=self.cal)
        self.actuator = actuator

    def execute(self, command: ActionCommand) -> None:
        """Apply the winning command's motion_vector to the base.

        Inactive commands and missing/zero vectors stop the wheels, so an
        idle arbitration result always leaves the robot halted.
        """
        if not command.active or command.motion_vector is None:
            self.stop()
            return

        v_x, _v_y, v_theta = command.motion_vector
        if v_x == 0 and v_theta == 0:
            self.stop()
            return

        left = v_x - v_theta
        right = v_x + v_theta

        # Renormalize instead of clamping so the left/right RATIO (the curve)
        # survives even when v_x + |v_theta| > 1.
        peak = max(1.0, abs(left), abs(right))
        left /= peak
        right /= peak

        # Pick the trim set the calibration was measured for.
        if v_x == 0:
            trim_set = "turn"
        elif v_x < 0:
            trim_set = "backward"
        else:
            trim_set = "forward"

        cmd = WheelCommand(
            left * self.cal.forward_speed,
            right * self.cal.forward_speed,
            True,
            trim_set,
        )
        self.actuator.apply(cmd)

    def stop(self) -> None:
        """Cut motor power (does not release GPIO)."""
        self.actuator.stop()

    def close(self) -> None:
        """Stop and release GPIO. Call once on shutdown."""
        self.actuator.close()
