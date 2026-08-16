from __future__ import annotations
from typing import Any, Optional
from src.subsumption.layers.base_layer import BaseLayer
from src.subsumption.arbitrator import ActionCommand
from src.hardware.sensors.sensor_hub import SensorHub

# Pivot fraction while steering away from a close obstacle -- independent of
# layer1_scan's TURN_SPEED on purpose (Rule 3: no cross-layer knowledge).
EMERGENCY_TURN_SPEED = 0.5


def _is_close(dist: Optional[float]) -> bool:
    return dist is not None and dist < SensorHub.EMERGENCY_STOP_CM


class EmergencyStopLayer(BaseLayer):
    """
    Layer 5: Emergency Stop
    Priority: 5 (High)
    Behavior: Suppresses all lower priority layers whenever any ultrasonic is
              inside EMERGENCY_STOP_CM. Steers away from whichever
              front-facing sensor tripped instead of just halting, so the
              robot doesn't sit dead until Layer 1 re-triggers a fresh scan
              cycle -- pure stop is reserved for cases with no safe steering
              direction: a rear obstacle (only relevant while backing up, and
              turning doesn't help you back up straight), or both front
              diagonals close at once (squeezed on both sides -- no direction
              is clearly safer).
    """
    def __init__(self):
        super().__init__(layer_id=5)

    def evaluate(self, sensors: Any) -> ActionCommand:
        if not sensors.has_obstacle():
            return ActionCommand(layer_id=self.layer_id, active=False)

        close_back = _is_close(sensors.get_obstacle_distance_back_cm())
        close_left = _is_close(sensors.get_obstacle_distance_front_left_cm())
        close_right = _is_close(sensors.get_obstacle_distance_front_right_cm())
        close_front = _is_close(sensors.get_obstacle_distance_cm())

        if close_back or (close_left and close_right):
            motion, message = (0, 0, 0), "EMERGENCY STOP"
        elif close_left:
            motion, message = (0, 0, -EMERGENCY_TURN_SPEED), "EMERGENCY TURN right (front_left blocked)"
        elif close_right:
            motion, message = (0, 0, EMERGENCY_TURN_SPEED), "EMERGENCY TURN left (front_right blocked)"
        elif close_front:
            motion, message = (0, 0, -EMERGENCY_TURN_SPEED), "EMERGENCY TURN right (front blocked)"
        else:
            motion, message = (0, 0, 0), "EMERGENCY STOP"

        return ActionCommand(
            layer_id=self.layer_id,
            active=True,
            motion_vector=motion,
            arm_action='stop',
            message=message,
        )