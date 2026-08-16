from __future__ import annotations
import time
from typing import Any, Optional
from src.subsumption.layers.base_layer import BaseLayer
from src.subsumption.arbitrator import ActionCommand
from src.hardware.sensors.sensor_hub import SensorHub

# Keep in step with layer1_scan's TURN_SPEED: this layer outvotes it, so a
# higher value here makes the robot speed up as it nears an obstacle.
EMERGENCY_TURN_SPEED = 0.2

# Release the turn only once everything is this much further than the
# trigger range. Releasing at the trigger range itself makes the robot
# oscillate on the threshold.
CLEAR_MARGIN = 1.6

# Once committed, hold the same direction at least this long. Re-deciding
# every tick is what made a single obstacle read left-then-right forever.
MIN_TURN_S = 0.6

# Give up turning after this long: still blocked all round means pivoting
# is not helping (wedged in a corner), so halt instead of spinning forever.
MAX_TURN_S = 6.0

_FAR = 1e6


def _dist(sensors: Any, getter: str) -> float:
    """None = no echo = nothing in range, so treat it as far, never as 0.
    Missing getter = sensor object predating the extra ultrasonics."""
    fn = getattr(sensors, getter, None)
    value = fn() if fn is not None else None
    return _FAR if value is None else value


class EmergencyStopLayer(BaseLayer):
    """
    Layer 5: Emergency Stop
    Priority: 5 (High)
    Behavior: Suppresses all lower priority layers whenever any ultrasonic is
              inside EMERGENCY_STOP_CM. Pivots away toward the roomier side
              and holds that direction until every sensor is clear, so the
              robot escapes instead of sitting dead or juddering in place.
              Halts instead of turning when turning cannot help: a rear
              obstacle, or no free side at all.
    """
    def __init__(self, turn_speed: float = EMERGENCY_TURN_SPEED):
        super().__init__(layer_id=5)
        self.turn_speed = EMERGENCY_TURN_SPEED
        self.set_turn_speed(turn_speed)
        self._turn_dir: Optional[float] = None   # +1 = left/CCW, -1 = right
        self._turn_started: float = 0.0

    def set_turn_speed(self, turn_speed: Optional[float] = None) -> None:
        """Pivot fraction 0..1, same contract as ScanAroundLayer.set_speeds."""
        if turn_speed is not None:
            self.turn_speed = max(0.0, min(1.0, float(turn_speed)))

    def evaluate(self, sensors: Any) -> ActionCommand:
        now = time.monotonic()
        front = _dist(sensors, "get_obstacle_distance_cm")
        back = _dist(sensors, "get_obstacle_distance_back_cm")
        left = _dist(sensors, "get_obstacle_distance_front_left_cm")
        right = _dist(sensors, "get_obstacle_distance_front_right_cm")

        trigger_cm = SensorHub.EMERGENCY_STOP_CM
        clear_cm = trigger_cm * CLEAR_MARGIN
        nearest = min(front, back, left, right)
        # Release looks at the forward sensors only: pivoting sweeps the rear
        # one past whatever we are escaping, and we resume driving forward.
        nearest_ahead = min(front, left, right)

        # Mid-turn: keep going until the way ahead clears the wider margin, so
        # the sensor that started this can't re-trigger on the boundary.
        if self._turn_dir is not None:
            turning_for = now - self._turn_started
            if turning_for >= MAX_TURN_S:
                self._turn_dir = None
                return self._command((0, 0, 0), "EMERGENCY STOP (turn timed out, wedged)")
            # MIN_TURN_S guards against a dropped echo (None reads as far)
            # ending the turn after a single tick.
            if turning_for >= MIN_TURN_S and nearest_ahead >= clear_cm:
                self._turn_dir = None
                return ActionCommand(layer_id=self.layer_id, active=False)
            return self._command(
                (0, 0, self._turn_dir * self.turn_speed),
                f"EMERGENCY TURN {'left' if self._turn_dir > 0 else 'right'} "
                f"(clearing, ahead {nearest_ahead:.0f}cm)",
            )

        if nearest >= trigger_cm:
            return ActionCommand(layer_id=self.layer_id, active=False)

        # Rear obstacle: pivoting does not open up space behind, and the only
        # way we get here is reversing, so just halt.
        if back < trigger_cm:
            return self._command((0, 0, 0), "EMERGENCY STOP (rear blocked)")

        # Turn toward the roomier diagonal rather than a fixed side -- a fixed
        # side turns back into the obstacle whenever it sits on that side.
        if max(left, right) < trigger_cm:
            return self._command((0, 0, 0), "EMERGENCY STOP (no free side)")

        self._turn_dir = 1.0 if right < left else -1.0
        self._turn_started = now
        if left >= trigger_cm and right >= trigger_cm:
            blocked = "front"
        else:
            blocked = "front_right" if right < left else "front_left"
        return self._command(
            (0, 0, self._turn_dir * self.turn_speed),
            f"EMERGENCY TURN {'left' if self._turn_dir > 0 else 'right'} ({blocked} blocked)",
        )

    def _command(self, motion, message: str) -> ActionCommand:
        return ActionCommand(
            layer_id=self.layer_id,
            active=True,
            motion_vector=motion,
            arm_action='stop',
            message=message,
        )
