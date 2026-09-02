"""
Layer 1: Scan Around — zigzag (boustrophedon) floor-coverage patrol.

No wheel encoders/IMU, so the base can't know where it is: it reacts to
the ultrasonic instead of following a map. Drive a lane until the front
sensor sees a wall, pivot ~90 deg, shift sideways ~one lane width, pivot
~90 deg again, alternating pivot side each wall so the path sweeps across
the room. A close obstacle (not a wall) gets dodged first; only a dodge
that fails to clear becomes a lane change. See src/scanning/phases.py for
the state machine and src/scanning/geometry.py for the sensor helpers.
"""
from __future__ import annotations

import random
import time
from typing import Any, Optional

from src.subsumption.layers.base_layer import BaseLayer
from src.subsumption.arbitrator import ActionCommand

from . import tuning
from . import geometry
from .phases import Phase, HANDLERS


def _target_locked(sensors: Any) -> bool:
    """Survives the occlusion grace window, unlike a bare get_litter_position() truthiness check."""
    getter = getattr(sensors, "get_litter_locked", None)
    if getter is not None:
        return bool(getter())
    return bool(sensors.get_litter_position())


class ScanAroundLayer(BaseLayer):
    """Priority 1 (very low). Zigzag patrol searching for targets; active
    only while no target is detected."""

    def __init__(self, forward_speed: float = tuning.FORWARD_SPEED,
                 turn_speed: float = tuning.TURN_SPEED,
                 turn_90_s: float = tuning.TURN_90_S,
                 shift_s: float = tuning.SHIFT_S,
                 max_lane_s: float = tuning.MAX_LANE_S):
        super().__init__(layer_id=1)
        self.set_speeds(forward_speed, turn_speed)
        self.set_timing(turn_90_s, shift_s, max_lane_s)
        self._phase: Optional[Phase] = None    # None = pattern not started
        self._phase_started: float = 0.0
        self._lane_started: float = 0.0
        self._lane_limit_s: float = self.max_lane_s
        self._turn_s: float = self.turn_90_s
        self._turn_left: bool = True           # pivot side; alternates per wall
        self._dodge_left: bool = True          # which way the current dodge went
        self._next_phase: Phase = Phase.DRIVE  # what the settle is settling for
        self._pivot_note: str = "L-- R--"
        self._suppressed_since: Optional[float] = None
        # SCAN-only mode has no Layer 2, so yielding would park the robot
        # on Layer 0 IDLE for as long as the camera could see a can.
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
        # Timers are open-loop; time spent suppressed by a higher layer
        # must not count against them.
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

        HANDLERS[self._phase](self, now, front, wall_ahead, sensors)

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
        """(v_x, v_y, v_theta) for the current phase. v_theta > 0 = CCW/left."""
        turn = self.turn_speed if self._turn_left else -self.turn_speed
        if self._phase == Phase.SETTLE:
            return (0, 0, 0), f"settling before {self._next_phase.name.lower()}"
        if self._phase == Phase.DRIVE:
            bias = geometry.lane_bias(left, right, self.turn_speed)
            if bias == 0.0:
                return (self.forward_speed, 0, 0), "lane"
            return ((self.forward_speed, 0, bias),
                    f"lane (nudge {bias:+.2f}, L{geometry.fmt(left)} R{geometry.fmt(right)})")
        dodge = self.turn_speed if self._dodge_left else -self.turn_speed
        dodge_side = 'left' if self._dodge_left else 'right'
        if self._phase == Phase.DODGE_TURN:
            return (0, 0, dodge), f"dodging {dodge_side} ({self._pivot_note})"
        if self._phase == Phase.DODGE_PASS:
            return (self.forward_speed, 0, 0), f"passing obstacle on the {'right' if self._dodge_left else 'left'}"
        if self._phase == Phase.DODGE_BACK:
            return (0, 0, -dodge), "back onto the lane"
        side = 'left' if self._turn_left else 'right'
        if self._phase == Phase.TURN1:
            return (0, 0, turn), f"turn 1 ({side}, v_theta {turn:+.2f}, {self._pivot_note})"
        if self._phase == Phase.SHIFT:
            return (self.forward_speed, 0, 0), "shifting lane"
        if self._phase == Phase.TURN2:
            return (0, 0, turn), f"turn 2 ({side}, v_theta {turn:+.2f}, {self._pivot_note})"
        return (0, 0, 0), "idle"

    def _settle(self, nxt: Phase, now: float) -> None:
        """Halt briefly before the wheels flip direction, then run `nxt`."""
        self._next_phase = nxt
        self._phase = Phase.SETTLE
        self._phase_started = now

    def _enter(self, phase: Phase, now: float) -> None:
        self._phase = phase
        self._phase_started = now
        if phase == Phase.TURN1:
            self._turn_s = geometry.jitter(self.turn_90_s)
        elif phase == Phase.TURN2:
            # Bounce: 90..180 deg, never less than 90 so the change completes.
            self._turn_s = self.turn_90_s * (1.0 + random.uniform(0.0, tuning.BOUNCE_EXTRA))

    def _start_lane(self, now: float) -> None:
        self._lane_started = now
        self._lane_limit_s = geometry.jitter(self.max_lane_s)

    def _elapsed(self, now: float) -> float:
        return now - self._phase_started

    def _start_dodge(self, sensors: Any, now: float) -> None:
        """Pivot toward whichever diagonal has more room. Never alternates
        for its own sake -- dodging into the tighter side wedges the robot
        into a corner."""
        left = geometry.side_room_cm(sensors, "get_obstacle_distance_front_left_cm")
        right = geometry.side_room_cm(sensors, "get_obstacle_distance_front_right_cm")
        self._pivot_note = f"L{left:.0f} R{right:.0f}"
        self._dodge_left = left >= right
        self._settle(Phase.DODGE_TURN, now)

    def _pivot_side(self, sensors: Any) -> bool:
        new_left, note = geometry.pivot_side(sensors, self._turn_left)
        self._pivot_note = note
        return new_left
