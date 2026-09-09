
from __future__ import annotations

import random
import time
from typing import Any, Optional

from src.subsumption.layers.base_layer import BaseLayer
from src.subsumption.arbitrator import ActionCommand

from src.scanning import tuning
from src.scanning import geometry
from src.scanning.phases import Phase, HANDLERS


def _target_locked(sensors: Any) -> bool:
    """True if a target is locked, surviving brief occlusion."""
    getter = getattr(sensors, "get_litter_locked", None)
    if getter is not None:
        return bool(getter())
    return bool(sensors.get_litter_position())


class ScanAroundLayer(BaseLayer):
    """Priority 1: zigzag patrol, active while no target is detected."""

    def __init__(self, forward_speed: float = tuning.FORWARD_SPEED,
                 turn_speed: float = tuning.TURN_SPEED,
                 turn_90_s: float = tuning.TURN_90_S,
                 shift_s: float = tuning.SHIFT_S,
                 max_lane_s: float = tuning.MAX_LANE_S):
        super().__init__(layer_id=1)
        self.set_speeds(forward_speed, turn_speed)
        self.set_timing(turn_90_s, shift_s, max_lane_s)
        self._phase: Optional[Phase] = None    # None = not started
        self._phase_started: float = 0.0
        self._lane_started: float = 0.0
        self._lane_limit_s: float = self.max_lane_s
        self._turn_s: float = self.turn_90_s
        self._turn_left: bool = True           # alternates per wall
        self._next_phase: Phase = Phase.DRIVE  # phase after settle
        self._pivot_note: str = "L-- R--"
        self._suppressed_since: Optional[float] = None
        # yielding needs Layer 2, else robot idles whenever camera sees a can
        self.yield_to_targets: bool = True

    # ── Calibration knobs (dashboard-tunable) ───────────────────────────────

    def set_timing(self, turn_90_s: Optional[float] = None,
                   shift_s: Optional[float] = None,
                   max_lane_s: Optional[float] = None) -> None:
        if turn_90_s is not None:
            self.turn_90_s = max(0.0, float(turn_90_s))
        if shift_s is not None:
            self.shift_s = max(0.0, float(shift_s))
        if max_lane_s is not None:
            self.max_lane_s = max(0.0, float(max_lane_s))

    def set_speeds(self, forward_speed: Optional[float] = None,
                   turn_speed: Optional[float] = None) -> None:
        if forward_speed is not None:
            self.forward_speed = max(0.0, min(1.0, float(forward_speed)))
        if turn_speed is not None:
            self.turn_speed = max(0.0, min(1.0, float(turn_speed)))

    def reset(self) -> None:
        self._phase = None
        self._suppressed_since = None

    def timing_summary(self) -> str:
        return (f"turn_90={self.turn_90_s:.2f}s shift={self.shift_s:.2f}s "
                f"max_lane={self.max_lane_s:.1f}s fwd={self.forward_speed:.2f} "
                f"turn={self.turn_speed:.2f}")

    # ── Subsumption API ──────────────────────────────────────────────────

    def notify_arbitration(self, won: bool) -> None:
        if not won and self._suppressed_since is None:
            self._suppressed_since = time.monotonic()

    def evaluate(self, sensors: Any) -> ActionCommand:
        if self.yield_to_targets and _target_locked(sensors):
            self._phase = None
            self._suppressed_since = None
            return ActionCommand(layer_id=self.layer_id, active=False)

        now = time.monotonic()
        if self._suppressed_since is not None:
            paused = now - self._suppressed_since
            self._phase_started += paused
            self._lane_started += paused
            self._suppressed_since = None

        front, left, right = geometry.clearances(sensors)
        wall_ahead = front is not None and front <= tuning.TURN_AT_CM

        if self._phase is None:
            self._enter(Phase.DRIVE, now)
            self._start_lane(now)

        HANDLERS[self._phase](self, now, front, wall_ahead, sensors) # type: ignore

        return self._output(front, left, right)

    # ── Internals ────────────────────────────────────────────────────────

    def _output(self, front, left, right) -> ActionCommand:
        motion, label = self._motion_for_phase(left, right)
        front_txt = f"{front:.0f}cm" if front is not None else "--"
        return ActionCommand(
            layer_id=self.layer_id,
            active=True,
            motion_vector=motion,
            arm_action='stow',
            message=f"SCAN {label} (front {front_txt})",
        )

    def _motion_for_phase(self, left=None, right=None):
        """Motion vector for the current phase; v_theta > 0 = left."""
        turn = self.turn_speed if self._turn_left else -self.turn_speed
        if self._phase == Phase.SETTLE:
            return (0, 0, 0), f"settling before {self._next_phase.name.lower()}"
        if self._phase == Phase.DRIVE:
            bias = geometry.lane_bias(left, right, self.turn_speed)
            if bias == 0.0:
                return (self.forward_speed, 0, 0), "lane"
            return ((self.forward_speed, 0, bias),
                    f"lane (nudge {bias:+.2f}, L{geometry.fmt(left)} R{geometry.fmt(right)})")
        side = 'left' if self._turn_left else 'right'
        if self._phase == Phase.TURN1:
            return (0, 0, turn), f"turn 1 ({side}, v_theta {turn:+.2f}, {self._pivot_note})"
        if self._phase == Phase.SHIFT:
            return (self.forward_speed, 0, 0), "shifting lane"
        if self._phase == Phase.TURN2:
            return (0, 0, turn), f"turn 2 ({side}, v_theta {turn:+.2f}, {self._pivot_note})"
        return (0, 0, 0), "idle"

    def _settle(self, nxt: Phase, now: float) -> None:
        """Halt briefly, then run `nxt`."""
        self._next_phase = nxt
        self._phase = Phase.SETTLE
        self._phase_started = now

    def _enter(self, phase: Phase, now: float) -> None:
        self._phase = phase
        self._phase_started = now
        if phase == Phase.TURN1:
            self._turn_s = geometry.jitter(self.turn_90_s)
        elif phase == Phase.TURN2:
            # bounce 90..180 deg, never less than 90
            self._turn_s = self.turn_90_s * (1.0 + random.uniform(0.0, tuning.BOUNCE_EXTRA))

    def _start_lane(self, now: float) -> None:
        self._lane_started = now
        self._lane_limit_s = geometry.jitter(self.max_lane_s)

    def _elapsed(self, now: float) -> float:
        return now - self._phase_started

    def _pivot_side(self, sensors: Any) -> bool:
        new_left, note = geometry.pivot_side(sensors, self._turn_left)
        self._pivot_note = note
        return new_left
