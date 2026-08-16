from __future__ import annotations
from typing import Any, Optional
from src.subsumption.layers.base_layer import BaseLayer
from src.subsumption.arbitrator import ActionCommand
from src.hardware.sensors.sensor_hub import SensorHub

# Keep in step with layer1_scan's TURN_SPEED: this layer outvotes it, so a
# higher value here makes the robot speed up as it nears an obstacle.
EMERGENCY_TURN_SPEED = 0.2


def _is_close(dist: Optional[float]) -> bool:
    return dist is not None and dist < SensorHub.EMERGENCY_STOP_CM


class EmergencyStopLayer(BaseLayer):
    """
    Layer 5: Emergency Stop
    Priority: 5 (High)
    Behavior: Suppresses all lower priority layers whenever any ultrasonic is
              inside EMERGENCY_STOP_CM. Steers away from the sensor that
              tripped; halts only when no direction is safer (rear obstacle,
              or both front diagonals blocked).
    """
    def __init__(self, turn_speed: float = EMERGENCY_TURN_SPEED):
        super().__init__(layer_id=5)
        self.turn_speed = EMERGENCY_TURN_SPEED
        self.set_turn_speed(turn_speed)

    def set_turn_speed(self, turn_speed: Optional[float] = None) -> None:
        """Pivot fraction 0..1, same contract as ScanAroundLayer.set_speeds."""
        if turn_speed is not None:
            self.turn_speed = max(0.0, min(1.0, float(turn_speed)))

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
            motion, message = (0, 0, -self.turn_speed), "EMERGENCY TURN right (front_left blocked)"
        elif close_right:
            motion, message = (0, 0, self.turn_speed), "EMERGENCY TURN left (front_right blocked)"
        elif close_front:
            motion, message = (0, 0, -self.turn_speed), "EMERGENCY TURN right (front blocked)"
        else:
            motion, message = (0, 0, 0), "EMERGENCY STOP"

        return ActionCommand(
            layer_id=self.layer_id,
            active=True,
            motion_vector=motion,
            arm_action='stop',
            message=message,
        )