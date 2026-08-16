from __future__ import annotations
import enum
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

# Reverse first so the chassis has room to pivot -- a rectangular base sweeps
# its corners wider than its front face, so "15cm ahead" is not 15cm of
# turning clearance. Only when the rear sensor sees at least this much space.
BACKOFF_S = 0.4
BACKOFF_CLEARANCE_CM = 30.0

# Brief halt between opposing wheel directions (safety doc: instant flips are
# current spikes that stall the driver and brown out the rail).
SETTLE_S = 0.15

# Diagonals within this much of each other carry no steering information.
TIE_MARGIN_CM = 5.0

_FAR = 1e6


def _dist(sensors: Any, getter: str) -> float:
    """None = no echo = nothing in range, so treat it as far, never as 0.
    Missing getter = sensor object predating the extra ultrasonics."""
    fn = getattr(sensors, getter, None)
    value = fn() if fn is not None else None
    return _FAR if value is None else value


class _Phase(enum.Enum):
    SETTLE_BACK = enum.auto()   # stop before the wheels flip to reverse
    BACKOFF     = enum.auto()
    SETTLE_TURN = enum.auto()   # stop before the wheels flip to pivot
    TURN        = enum.auto()
    WEDGED      = enum.auto()   # gave up; hold still until something clears


class EmergencyStopLayer(BaseLayer):
    """
    Layer 5: Emergency Stop
    Priority: 5 (High)
    Behavior: Suppresses all lower priority layers whenever any ultrasonic is
              inside EMERGENCY_STOP_CM. Backs off (when the rear is clear),
              then pivots toward the roomier side and holds that direction
              until the way ahead is clear, so the robot escapes instead of
              sitting dead or juddering in place. Halts when nothing helps:
              a rear-only obstacle, or blocked ahead with no room behind.
    """
    def __init__(self, turn_speed: float = EMERGENCY_TURN_SPEED):
        super().__init__(layer_id=5)
        self.turn_speed = EMERGENCY_TURN_SPEED
        self.set_turn_speed(turn_speed)
        self._phase: Optional[_Phase] = None
        self._phase_started: float = 0.0
        self._turn_dir: float = -1.0             # +1 = left/CCW, -1 = right
        self._tie_dir: float = -1.0              # side to try when neither diagonal is nearer

    def set_turn_speed(self, turn_speed: Optional[float] = None) -> None:
        """Pivot fraction 0..1, same contract as ScanAroundLayer.set_speeds.
        Doubles as the backoff speed so there is one knob, not two."""
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
        elapsed = now - self._phase_started

        if self._phase == _Phase.SETTLE_BACK:
            if elapsed < SETTLE_S:
                return self._command((0, 0, 0), "EMERGENCY settling before backoff")
            self._enter(_Phase.BACKOFF, now)
            return self._command((-self.turn_speed, 0, 0),
                                  f"EMERGENCY BACKOFF (rear {back:.0f}cm)")

        if self._phase == _Phase.BACKOFF:
            # Abort early if something turns up behind us mid-reverse.
            if elapsed >= BACKOFF_S or back < trigger_cm:
                self._enter(_Phase.SETTLE_TURN, now)
                return self._command((0, 0, 0), "EMERGENCY settling before turn")
            return self._command((-self.turn_speed, 0, 0),
                                  f"EMERGENCY BACKOFF (rear {back:.0f}cm)")

        if self._phase == _Phase.SETTLE_TURN:
            if elapsed >= SETTLE_S:
                self._turn_dir = self._pick_direction(left, right)
                self._enter(_Phase.TURN, now)
                return self._turn_command(front, left, right, trigger_cm)
            return self._command((0, 0, 0), "EMERGENCY settling before turn")

        if self._phase == _Phase.WEDGED:
            # Latched: pivoting already failed once, so re-deciding every tick
            # just restarts the same doomed manoeuvre forever. Only a genuine
            # change in the readings releases it.
            if min(front, left, right) >= clear_cm:
                self._phase = None
                return ActionCommand(layer_id=self.layer_id, active=False)
            return self._command((0, 0, 0), "EMERGENCY STOP (wedged, turning did not help)")

        if self._phase == _Phase.TURN:
            if elapsed >= MAX_TURN_S:
                self._enter(_Phase.WEDGED, now)
                return self._command((0, 0, 0), "EMERGENCY STOP (wedged, turning did not help)")
            # MIN_TURN_S guards against a dropped echo (None reads as far)
            # ending the turn after a single tick.
            ahead = min(front, left, right)
            if elapsed >= MIN_TURN_S and ahead >= clear_cm:
                self._phase = None
                return ActionCommand(layer_id=self.layer_id, active=False)
            return self._command(
                (0, 0, self._turn_dir * self.turn_speed),
                f"EMERGENCY TURN {'left' if self._turn_dir > 0 else 'right'} "
                f"(clearing, ahead {ahead:.0f}cm)",
            )

        # ---- no manoeuvre running: decide whether to start one -------------
        if min(front, back, left, right) >= trigger_cm:
            return ActionCommand(layer_id=self.layer_id, active=False)

        if min(front, left, right) >= trigger_cm:
            # Only the rear is blocked, and pivoting opens no space behind.
            return self._command((0, 0, 0), "EMERGENCY STOP (rear blocked)")

        if back >= BACKOFF_CLEARANCE_CM:
            self._enter(_Phase.SETTLE_BACK, now)
            return self._command((0, 0, 0), "EMERGENCY settling before backoff")

        if left < trigger_cm and right < trigger_cm:
            # Blocked both sides with no room to reverse into.
            return self._command((0, 0, 0), "EMERGENCY STOP (boxed in)")

        self._enter(_Phase.SETTLE_TURN, now)
        return self._command((0, 0, 0), "EMERGENCY settling before turn")

    def _pick_direction(self, left: float, right: float) -> float:
        """Turn toward the roomier diagonal. Both sensors reading nothing (an
        open field of view gives no echo) is a tie, not a reason to always
        pick the same side -- alternate, so a side that failed last time is
        not retried forever."""
        if abs(left - right) < TIE_MARGIN_CM:
            self._tie_dir = -self._tie_dir
            return self._tie_dir
        return 1.0 if right < left else -1.0

    def _turn_command(self, front, left, right, trigger_cm) -> ActionCommand:
        if left >= trigger_cm and right >= trigger_cm:
            blocked = "front"
        else:
            blocked = "front_right" if right < left else "front_left"
        return self._command(
            (0, 0, self._turn_dir * self.turn_speed),
            f"EMERGENCY TURN {'left' if self._turn_dir > 0 else 'right'} ({blocked} blocked)",
        )

    def _enter(self, phase: _Phase, now: float) -> None:
        self._phase = phase
        self._phase_started = now

    def _command(self, motion, message: str) -> ActionCommand:
        return ActionCommand(
            layer_id=self.layer_id,
            active=True,
            motion_vector=motion,
            arm_action='stop',
            message=message,
        )
