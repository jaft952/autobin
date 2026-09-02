from __future__ import annotations
import enum
import time
from typing import Any, Callable, NamedTuple, Optional
from src.subsumption.layers.base_layer import BaseLayer
from src.subsumption.arbitrator import ActionCommand
from src.hardware.sensors.sensor_hub import SensorHub
from src.hardware.sensors.clearance import read_distance_cm
from src.safety.obstacle_avoidance import compare_room

EMERGENCY_TURN_SPEED = 0.2   # keep in step with src.scanning.tuning.TURN_SPEED, this layer outvotes it
CLEAR_MARGIN = 1.6           # release the turn only once past this multiple of the trigger range
DIAG_CLEAR_MARGIN = 1.2      # diagonals only need to leave the trigger range, not the full front margin
MIN_TURN_S = 0.6             # hold a turn direction at least this long, re-deciding every tick oscillated
MAX_TURN_S = 6.0             # still blocked after this long -> edging away isn't working, escape instead

# Wedge escape: reverse, spin about-face, drive out, retrying wider each time. No give-up state.
TURN_180_S     = 2.0
SPIN_STEP_S    = 0.5
MAX_SPIN_S     = 4.0
ESCAPE_BACK_S  = 1.0
ESCAPE_DRIVE_S = 1.0

BACKOFF_S = 0.4              # reverse first for turning room, a rectangular base sweeps its corners wide
BACKOFF_CLEARANCE_CM = 30.0
SETTLE_S = 0.15              # halt between opposing wheel directions, instant flips brown out the rail
TIE_MARGIN_CM = 5.0          # diagonals within this much of each other carry no steering information
ROOMY_CM = 40.0              # past this range a diagonal counts as open regardless of exact distance

_FAR = 1e6


def _dist(sensors: Any, getter: str) -> float:
    """None (no echo / missing getter) means nothing in range, never 0."""
    value = read_distance_cm(sensors, getter)
    return _FAR if value is None else value


class _Phase(enum.Enum):
    SETTLE  = enum.auto()
    BACKOFF = enum.auto()
    TURN    = enum.auto()
    SPIN    = enum.auto()
    NUDGE   = enum.auto()


class _Readings(NamedTuple):
    front: float
    back: float
    left: float
    right: float
    trigger_cm: float
    now: float


class EmergencyStopLayer(BaseLayer):
    """Priority 5. Suppresses everything below EMERGENCY_STOP_CM: backs off, pivots to the roomier side, or wedge-escapes."""

    def __init__(self, turn_speed: float = EMERGENCY_TURN_SPEED,
                 grab_zone_check: Optional[Callable[[Any], bool]] = None):
        super().__init__(layer_id=5)
        self.turn_speed = EMERGENCY_TURN_SPEED
        self.set_turn_speed(turn_speed)
        self.grab_zone_check = grab_zone_check   # e.g. CollectLitterLayer.is_grabbable; exempts the tin being grabbed
        self._phase: Optional[_Phase] = None
        self._phase_started: float = 0.0
        self._turn_dir: float = -1.0
        self._tie_dir: float = -1.0
        self._next_phase: _Phase = _Phase.TURN
        self._label: str = "turn"
        self._after_backoff: _Phase = _Phase.TURN
        self._backoff_s: float = BACKOFF_S
        self._attempt: int = 0
        self._spin_s: float = TURN_180_S

    def set_turn_speed(self, turn_speed: Optional[float] = None) -> None:
        if turn_speed is not None:
            self.turn_speed = max(0.0, min(1.0, float(turn_speed)))

    def reset(self) -> None:
        self._reset()

    _PHASE_HANDLERS = {
        _Phase.SETTLE:  "_evaluate_settle",
        _Phase.BACKOFF: "_evaluate_backoff",
        _Phase.TURN:    "_evaluate_turn",
        _Phase.SPIN:    "_evaluate_spin",
        _Phase.NUDGE:   "_evaluate_nudge",
    }

    def evaluate(self, sensors: Any) -> ActionCommand:
        r = self._read(sensors)
        handler_name = self._PHASE_HANDLERS.get(self._phase) if self._phase else None
        if handler_name is not None:
            return getattr(self, handler_name)(r)
        return self._decide_new_manoeuvre(sensors, r)

    @staticmethod
    def _read(sensors: Any) -> _Readings:
        return _Readings(
            front=_dist(sensors, "get_obstacle_distance_cm"),
            back=_dist(sensors, "get_obstacle_distance_back_cm"),
            left=_dist(sensors, "get_obstacle_distance_front_left_cm"),
            right=_dist(sensors, "get_obstacle_distance_front_right_cm"),
            trigger_cm=SensorHub.EMERGENCY_STOP_CM,
            now=time.monotonic(),
        )

    # ── Phase handlers ───────────────────────────────────────────────────

    def _evaluate_settle(self, r: _Readings) -> ActionCommand:
        elapsed = r.now - self._phase_started
        if elapsed < SETTLE_S:
            return self._command((0, 0, 0), f"EMERGENCY settling before {self._label}")
        self._enter(self._next_phase, r.now)
        if self._phase == _Phase.TURN:
            self._turn_dir = self._pick_direction(r.left, r.right)
        elif self._phase == _Phase.NUDGE and r.front < r.trigger_cm:
            self._phase = None
            return ActionCommand(layer_id=self.layer_id, active=False)
        return self._phase_command(r.front, r.back, r.left, r.right, r.trigger_cm)

    def _evaluate_backoff(self, r: _Readings) -> ActionCommand:
        elapsed = r.now - self._phase_started
        if elapsed >= self._backoff_s or r.back < r.trigger_cm:
            return self._settle(self._after_backoff, r.now)
        return self._phase_command(r.front, r.back, r.left, r.right, r.trigger_cm)

    def _evaluate_turn(self, r: _Readings) -> ActionCommand:
        elapsed = r.now - self._phase_started
        if elapsed >= MAX_TURN_S:
            return self._start_escape(r.back, r.now)
        clear_cm = r.trigger_cm * CLEAR_MARGIN
        diag_clear_cm = r.trigger_cm * DIAG_CLEAR_MARGIN
        escaped = r.front >= clear_cm and min(r.left, r.right) >= diag_clear_cm
        if elapsed >= MIN_TURN_S and escaped:
            self._reset()
            return ActionCommand(layer_id=self.layer_id, active=False)
        return self._phase_command(r.front, r.back, r.left, r.right, r.trigger_cm)

    def _evaluate_spin(self, r: _Readings) -> ActionCommand:
        elapsed = r.now - self._phase_started
        if elapsed >= self._spin_s:
            return self._settle(_Phase.NUDGE, r.now)
        return self._phase_command(r.front, r.back, r.left, r.right, r.trigger_cm)

    def _evaluate_nudge(self, r: _Readings) -> ActionCommand:
        elapsed = r.now - self._phase_started
        if elapsed >= ESCAPE_DRIVE_S or r.front < r.trigger_cm:
            self._phase = None
            return ActionCommand(layer_id=self.layer_id, active=False)
        return self._phase_command(r.front, r.back, r.left, r.right, r.trigger_cm)

    def _decide_new_manoeuvre(self, sensors: Any, r: _Readings) -> ActionCommand:
        """The rear only gates reversing here, never triggers a manoeuvre on its own."""
        if min(r.front, r.left, r.right) >= r.trigger_cm:
            self._reset()
            return ActionCommand(layer_id=self.layer_id, active=False)

        if self._stand_down_for_grab(sensors):
            self._reset()
            return ActionCommand(layer_id=self.layer_id, active=False)

        if r.left < r.trigger_cm and r.right < r.trigger_cm and r.back < BACKOFF_CLEARANCE_CM:
            return self._start_escape(r.back, r.now)

        if r.back >= BACKOFF_CLEARANCE_CM:
            self._backoff_s = BACKOFF_S
            self._after_backoff = _Phase.TURN
            return self._settle(_Phase.BACKOFF, r.now)

        return self._settle(_Phase.TURN, r.now)

    # ── Escape ───────────────────────────────────────────────────────────

    def _start_escape(self, back: float, now: float) -> ActionCommand:
        self._attempt += 1
        self._spin_s = min(TURN_180_S + SPIN_STEP_S * (self._attempt - 1), MAX_SPIN_S)
        if back >= BACKOFF_CLEARANCE_CM:
            self._backoff_s = ESCAPE_BACK_S
            self._after_backoff = _Phase.SPIN
            return self._settle(_Phase.BACKOFF, now)
        return self._settle(_Phase.SPIN, now)

    def _stand_down_for_grab(self, sensors: Any) -> bool:
        """A broken predicate must never disarm the e-stop -- fail safe."""
        if self.grab_zone_check is None:
            return False
        try:
            return bool(self.grab_zone_check(sensors))
        except Exception:
            return False

    def _pick_direction(self, left: float, right: float) -> float:
        favors_left = compare_room(left, right, roomy_cm=ROOMY_CM,
                                    tie_margin_cm=TIE_MARGIN_CM)
        if favors_left is None:
            self._tie_dir = -self._tie_dir
            return self._tie_dir
        return 1.0 if favors_left else -1.0

    def _phase_command(self, front, back, left, right, trigger_cm) -> ActionCommand:
        speed = self.turn_speed
        if self._phase == _Phase.BACKOFF:
            return self._command((-speed, 0, 0), f"EMERGENCY BACKOFF (rear {back:.0f}cm)")
        if self._phase == _Phase.SPIN:
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
        self._next_phase = nxt
        self._label = nxt.name.lower()
        self._enter(_Phase.SETTLE, now)
        return self._command((0, 0, 0), f"EMERGENCY settling before {self._label}")

    def _enter(self, phase: _Phase, now: float) -> None:
        self._phase = phase
        self._phase_started = now

    def _reset(self) -> None:
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
