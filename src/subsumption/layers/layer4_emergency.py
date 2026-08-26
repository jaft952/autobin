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

# The diagonals only have to leave the trigger range, not clear the full front
# margin: in a corridor the side walls never reach 1.6x, so demanding it of
# them meant the escape always timed out into WEDGED.
DIAG_CLEAR_MARGIN = 1.2

# Once committed, hold the same direction at least this long. Re-deciding
# every tick is what made a single obstacle read left-then-right forever.
MIN_TURN_S = 0.6

# Give up turning after this long: still blocked all round means edging away
# is not working, so run the escape below instead.
MAX_TURN_S = 6.0

# --- Wedge escape: reverse, spin about-face, drive out, repeat -------------
# Each retry spins further than the last, so the robot stops retracing the
# same arc back into the same corner. There is deliberately no give-up state:
# stopping dead in a corner needs a human to come and lift the robot.
TURN_180_S     = 2.0   # half a turn clockwise; ~2x ScanAroundLayer.TURN_90_S
SPIN_STEP_S    = 0.5   # added per retry
MAX_SPIN_S     = 4.0
ESCAPE_BACK_S  = 1.0   # longer reverse than the routine backoff
ESCAPE_DRIVE_S = 1.0   # forward burst that actually leaves the spot

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

# Past this range a diagonal is "open", however far it actually reads. A wall
# 60cm off one side is not a reason to steer away from it, but comparing raw
# distances made it outrank a side reading nothing at all.
ROOMY_CM = 40.0

_FAR = 1e6


def _dist(sensors: Any, getter: str) -> float:
    """None = no echo = nothing in range, so treat it as far, never as 0.
    Missing getter = sensor object predating the extra ultrasonics."""
    fn = getattr(sensors, getter, None)
    value = fn() if fn is not None else None
    return _FAR if value is None else value


class _Phase(enum.Enum):
    SETTLE  = enum.auto()   # stop before the wheels flip direction
    BACKOFF = enum.auto()
    TURN    = enum.auto()
    SPIN    = enum.auto()   # wedged: about-face clockwise
    NUDGE   = enum.auto()   # drive out of the spot the spin turned away from


class EmergencyStopLayer(BaseLayer):
    """
    Layer 5: Emergency Stop
    Priority: 5 (High)
    Behavior: Suppresses all lower priority layers whenever any ultrasonic is
              inside EMERGENCY_STOP_CM. Backs off (when the rear is clear),
              then pivots toward the roomier side and holds that direction
              until the way ahead is clear. When edging away gets nowhere it
              reverses, spins about-face clockwise and drives out, retrying
              with a wider spin each time -- it never latches into a stop,
              because a robot stopped in a corner needs a human to free it.
              The rear sensor only gates reversing; it never triggers a stop
              on its own.
    """
    def __init__(self, turn_speed: float = EMERGENCY_TURN_SPEED):
        super().__init__(layer_id=5)
        self.turn_speed = EMERGENCY_TURN_SPEED
        self.set_turn_speed(turn_speed)
        self._phase: Optional[_Phase] = None
        self._phase_started: float = 0.0
        self._turn_dir: float = -1.0             # +1 = left/CCW, -1 = right
        self._tie_dir: float = -1.0              # side to try when neither diagonal is nearer
        self._next_phase: _Phase = _Phase.TURN   # what the settle is settling for
        self._label: str = "turn"
        self._after_backoff: _Phase = _Phase.TURN
        self._backoff_s: float = BACKOFF_S
        self._attempt: int = 0                   # escape retries since last clear
        self._spin_s: float = TURN_180_S

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
        diag_clear_cm = trigger_cm * DIAG_CLEAR_MARGIN
        escaped = front >= clear_cm and min(left, right) >= diag_clear_cm
        elapsed = now - self._phase_started

        if self._phase == _Phase.SETTLE:
            if elapsed < SETTLE_S:
                return self._command((0, 0, 0), f"EMERGENCY settling before {self._label}")
            self._enter(self._next_phase, now)
            if self._phase == _Phase.TURN:
                self._turn_dir = self._pick_direction(left, right)
            elif self._phase == _Phase.NUDGE and front < trigger_cm:
                # The spin did not open a way out: hand back and let the next
                # tick start a fresh, wider attempt.
                self._phase = None
                return ActionCommand(layer_id=self.layer_id, active=False)
            return self._phase_command(front, back, left, right, trigger_cm)

        if self._phase == _Phase.BACKOFF:
            # Abort early if something turns up behind us mid-reverse.
            if elapsed >= self._backoff_s or back < trigger_cm:
                return self._settle(self._after_backoff, now)
            return self._phase_command(front, back, left, right, trigger_cm)

        if self._phase == _Phase.TURN:
            if elapsed >= MAX_TURN_S:
                return self._start_escape(back, now)
            # MIN_TURN_S guards against a dropped echo (None reads as far)
            # ending the turn after a single tick.
            if elapsed >= MIN_TURN_S and escaped:
                self._reset()
                return ActionCommand(layer_id=self.layer_id, active=False)
            return self._phase_command(front, back, left, right, trigger_cm)

        if self._phase == _Phase.SPIN:
            # Run the about-face to completion: stopping the moment a sensor
            # reads clear leaves the robot half way round, still facing the
            # corner it is trying to leave.
            if elapsed >= self._spin_s:
                return self._settle(_Phase.NUDGE, now)
            return self._phase_command(front, back, left, right, trigger_cm)

        if self._phase == _Phase.NUDGE:
            # Turning away is not leaving. Without this the robot faced a way
            # out, handed back to the patrol, and was re-triggered on the spot.
            if elapsed >= ESCAPE_DRIVE_S or front < trigger_cm:
                self._phase = None
                return ActionCommand(layer_id=self.layer_id, active=False)
            return self._phase_command(front, back, left, right, trigger_cm)

        # ---- no manoeuvre running: decide whether to start one -------------
        # The rear never triggers a manoeuvre: it only gates whether reversing
        # is allowed (below) and aborts a backoff already in progress. Driving
        # forward past something behind the robot is not an emergency, and
        # halting for it stranded the patrol against walls it had left.
        if min(front, left, right) >= trigger_cm:
            self._reset()
            return ActionCommand(layer_id=self.layer_id, active=False)

        if left < trigger_cm and right < trigger_cm and back < BACKOFF_CLEARANCE_CM:
            # Blocked both sides with no room to reverse into: edging away has
            # nowhere to go, so skip straight to the about-face.
            return self._start_escape(back, now)

        if back >= BACKOFF_CLEARANCE_CM:
            self._backoff_s = BACKOFF_S
            self._after_backoff = _Phase.TURN
            return self._settle(_Phase.BACKOFF, now)

        return self._settle(_Phase.TURN, now)

    # ── Escape ────────────────────────────────────────────────────────────

    def _start_escape(self, back: float, now: float) -> ActionCommand:
        """Edging away got nowhere. Reverse if there is room, spin about-face,
        then drive out. Each retry spins further so the robot stops retracing
        the same arc; there is no give-up state."""
        self._attempt += 1
        self._spin_s = min(TURN_180_S + SPIN_STEP_S * (self._attempt - 1), MAX_SPIN_S)
        if back >= BACKOFF_CLEARANCE_CM:
            self._backoff_s = ESCAPE_BACK_S
            self._after_backoff = _Phase.SPIN
            return self._settle(_Phase.BACKOFF, now)
        return self._settle(_Phase.SPIN, now)

    def _pick_direction(self, left: float, right: float) -> float:
        """Turn toward the roomier diagonal, judged on obstruction rather than
        raw range: anything past ROOMY_CM counts as equally open, so a distant
        wall on one side no longer loses to a side that simply got no echo.
        A genuine tie alternates, so a side that failed last time is not
        retried forever."""
        near_left = min(left, ROOMY_CM)
        near_right = min(right, ROOMY_CM)
        if abs(near_left - near_right) < TIE_MARGIN_CM:
            self._tie_dir = -self._tie_dir
            return self._tie_dir
        return 1.0 if near_right < near_left else -1.0

    def _phase_command(self, front, back, left, right, trigger_cm) -> ActionCommand:
        """The motion the current phase drives, plus the readings behind it."""
        speed = self.turn_speed
        if self._phase == _Phase.BACKOFF:
            return self._command((-speed, 0, 0), f"EMERGENCY BACKOFF (rear {back:.0f}cm)")
        if self._phase == _Phase.SPIN:
            # Clockwise is -ve v_theta (see motion_executor).
            return self._command(
                (0, 0, -abs(speed)),
                f"EMERGENCY SPIN clockwise (try {self._attempt}, {self._spin_s:.1f}s)")
        if self._phase == _Phase.NUDGE:
            return self._command((speed, 0, 0), f"EMERGENCY DRIVE OUT (front {front:.0f}cm)")
        v_theta = self._turn_dir * speed
        return self._command(
            (0, 0, v_theta),
            f"EMERGENCY TURN {'left' if v_theta > 0 else 'right'} "
            f"(v_theta {v_theta:+.2f}, L{min(left, ROOMY_CM):.0f} "
            f"R{min(right, ROOMY_CM):.0f}, ahead {min(front, left, right):.0f}cm)",
        )

    def _settle(self, nxt: _Phase, now: float) -> ActionCommand:
        """Brief halt before the wheels flip direction, then run `nxt`."""
        self._next_phase = nxt
        self._label = nxt.name.lower()
        self._enter(_Phase.SETTLE, now)
        return self._command((0, 0, 0), f"EMERGENCY settling before {self._label}")

    def _enter(self, phase: _Phase, now: float) -> None:
        self._phase = phase
        self._phase_started = now

    def _reset(self) -> None:
        """Genuinely clear again, so the next wedge escalates from scratch."""
        self._phase = None
        self._attempt = 0

    def _command(self, motion, message: str) -> ActionCommand:
        return ActionCommand(
            layer_id=self.layer_id,
            active=True,
            motion_vector=motion,
            arm_action='stop',
            message=message,
        )
