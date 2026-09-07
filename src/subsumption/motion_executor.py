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

HALTING: a zero motion_vector COASTS the wheels -- all four inputs LOW, so
each motor is open-circuit and free to spin. There is no brake.

There used to be one: both inputs of each motor driven HIGH shorts the
windings, which resists being pushed, and a grab used it so the arm's
shaking could not drift the base off its aligned spot. On this hardware it
did the opposite. Held that way the base crept and yawed for a whole run of
all-HIGH ticks, while a coasting base sat still. Four independently
soft-timed PWM channels held at "100%" are not a guaranteed solid HIGH on
all four at once, and any moment where one input of a motor is high while
its partner is not is a DRIVE pulse -- per channel, so it steers as well as
creeps. Removed rather than left as a trap; see
docs/hardware_safety_patterns.md section 8.
"""
from __future__ import annotations

import math
import time
from typing import Optional

from src.hardware.actuators.interfaces import ActuatorInterface
from src.motion.calibration import MotionCalibration, MotorPins
from src.motion.differential_kinematics import WheelCommand
from src.subsumption.arbitrator import ActionCommand

# SLEW LIMIT: how fast the driven vector may CHANGE, in vector units per
# SECOND. Per second, not per tick: the loop rate is not constant, so a
# per-tick cap would mean a different thing at every rate.
#
# A layer that wins arbitration used to take effect whole on the very next
# tick. Layer 1 driving straight, then Layer 2 taking over with a 79-degree
# arc, is a one-tick jump from (0.2, 0, 0) to (0.25, 0, 0.10): one wheel
# +78%, the other -26%, and the chassis snaps sideways. Detection noise does
# the same inside Layer 2, where the steer angle is 180x the lateral error.
#
# Ramping applies to speeding up ONLY. Slowing, stopping and reversing all
# take effect at once -- a stop that arrives late is a safety bug, and a
# direction flip is forced THROUGH zero rather than eased through it
# (hardware_safety_patterns.md rule 7).
SLEW_VX_PER_S = 0.6
SLEW_VTHETA_PER_S = 0.3

_SLEW_MAX_DT_S = 0.5      # a long stall must not authorise an unlimited step


def _slew(current: float, target: float, max_step: float) -> float:
    """One component, moved toward `target` by at most `max_step`."""
    if target == 0.0:
        return 0.0                       # stopping is never delayed
    if current != 0.0 and (current > 0.0) != (target > 0.0):
        return 0.0                       # direction flip passes through zero
    if abs(target) <= abs(current):
        return target                    # slowing down is free
    delta = target - current
    if abs(delta) <= max_step:
        return target
    return current + math.copysign(max_step, delta)


class MotionExecutor:
    """Executes the winning ActionCommand on the wheels. One per robot."""

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
        self._vec = (0.0, 0.0)          # (v_x, v_theta) actually being driven
        self._vec_at: Optional[float] = None

    def execute(self, command: ActionCommand) -> None:
        """Apply the winning command's motion_vector to the base.

        Inactive commands and missing/zero vectors stop the wheels, so an
        idle arbitration result always leaves the robot halted.
        """
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

    def _ramp(self, v_x: float, v_theta: float):
        """Limit how far the driven vector may move since the last call."""
        now = time.monotonic()
        dt = _SLEW_MAX_DT_S if self._vec_at is None else min(now - self._vec_at,
                                                            _SLEW_MAX_DT_S)
        self._vec_at = now
        cur_x, cur_theta = self._vec
        self._vec = (_slew(cur_x, v_x, SLEW_VX_PER_S * dt),
                     _slew(cur_theta, v_theta, SLEW_VTHETA_PER_S * dt))
        return self._vec

    def _halted(self) -> None:
        """The wheels are not turning, so the ramp restarts from rest."""
        self._vec = (0.0, 0.0)
        self._vec_at = time.monotonic()

    def stop(self) -> None:
        """Coast: cut motor power, wheels free to spin (does not release GPIO)."""
        self._halted()
        self.actuator.stop()

    def close(self) -> None:
        """Stop and release GPIO. Call once on shutdown."""
        self.actuator.close()
