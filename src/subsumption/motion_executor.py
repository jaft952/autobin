"""Runs the winning movement command on the wheels."""
from __future__ import annotations

import math
import time
from typing import Optional

from src.hardware.actuators.interfaces import ActuatorInterface
from src.motion.calibration import MotionCalibration, MotorPins
from src.motion.differential_kinematics import WheelCommand
from src.subsumption.arbitrator import ActionCommand

SLEW_VX_PER_S = 0.6
SLEW_VTHETA_PER_S = 0.3

_SLEW_MAX_DT_S = 0.5


def _slew(current: float, target: float, max_step: float) -> float:
    """Move a value toward the target by a limited step."""
    if target == 0.0:
        return 0.0
    if current != 0.0 and (current > 0.0) != (target > 0.0):
        return 0.0
    if abs(target) <= abs(current):
        return target
    delta = target - current
    if abs(delta) <= max_step:
        return target
    return current + math.copysign(max_step, delta)


class MotionExecutor:
    """Carries out the movement part of the winning command."""

    def __init__(
        self,
        actuator: Optional[ActuatorInterface] = None,
        calibration: Optional[MotionCalibration] = None,
        pins: Optional[MotorPins] = None,
    ) -> None:
        self.cal = calibration or MotionCalibration()
        if actuator is None:
            from src.hardware.actuators.pwm_driver import PWMActuator
            actuator = PWMActuator(pins=pins, calibration=self.cal)
        self.actuator: ActuatorInterface = actuator
        self._vec = (0.0, 0.0)
        self._vec_at: Optional[float] = None

    def execute(self, command: ActionCommand) -> None:
        """Drive the wheels with the winning command."""
        if not command.active or command.motion_vector is None:
            self.stop()
            return

        v_x, _v_y, v_theta = command.motion_vector
        v_x, v_theta = self._ramp(v_x, v_theta)
        if v_x == 0 and v_theta == 0:
            self.stop()
            return

        left = v_x - v_theta
        right = v_x + v_theta

        peak = max(1.0, abs(left), abs(right))
        left /= peak
        right /= peak

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

    def _ramp(self, v_x: float, v_theta: float):
        """Limit how fast the speed can change."""
        now = time.monotonic()
        dt = _SLEW_MAX_DT_S if self._vec_at is None else min(now - self._vec_at,
                                                            _SLEW_MAX_DT_S)
        self._vec_at = now
        cur_x, cur_theta = self._vec
        self._vec = (_slew(cur_x, v_x, SLEW_VX_PER_S * dt),
                     _slew(cur_theta, v_theta, SLEW_VTHETA_PER_S * dt))
        return self._vec

    def _ramp_reset(self) -> None:
        """Restart the speed ramp from zero."""
        self._vec = (0.0, 0.0)
        self._vec_at = time.monotonic()

    def stop(self) -> None:
        """Cut motor power and let the wheels roll."""
        self._ramp_reset()
        self.actuator.stop()

    def close(self) -> None:
        """Stop and release the GPIO pins."""
        self.actuator.close()
