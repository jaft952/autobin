"""MotionExecutor: turns the winning ActionCommand's motion_vector into wheel motion (mix, renormalize, brake vs coast on zero)."""
from __future__ import annotations

from typing import Optional

from src.hardware.actuators.interfaces import ActuatorInterface
from src.motion.calibration import MotionCalibration, MotorPins
from src.motion.differential_kinematics import WheelCommand
from src.subsumption.arbitrator import ActionCommand

# arm_actions where a zero vector must brake (hold), not coast.
_GRAB_ACTIONS = frozenset({"grab_arc", "grab_sequence", "hold", "deploy"})


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

    def execute(self, command: ActionCommand) -> None:
        """Apply motion_vector to the base; inactive/missing/zero vectors stop the wheels."""
        if not command.active or command.motion_vector is None:
            self.stop()
            return

        v_x, _v_y, v_theta = command.motion_vector
        if v_x == 0 and v_theta == 0:
            # Hold position (brake) while the arm grabs; coast otherwise.
            if command.arm_action in _GRAB_ACTIONS:
                self.brake()
            else:
                self.stop()
            return

        left = v_x - v_theta
        right = v_x + v_theta

        # Renormalize (not clamp) so the left/right ratio survives overrange.
        peak = max(1.0, abs(left), abs(right))
        left /= peak
        right /= peak

        # Pick the calibration trim set for this direction.
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
        """Coast: cut motor power, wheels free to spin (does not release GPIO)."""
        self.actuator.stop()

    def brake(self) -> None:
        """Actively hold position during a grab; falls back to stop() if unsupported."""
        brake_fn = getattr(self.actuator, "brake", None)
        if callable(brake_fn):
            brake_fn()
        else:
            self.actuator.stop()

    def close(self) -> None:
        """Stop and release GPIO. Call once on shutdown."""
        self.actuator.close()
