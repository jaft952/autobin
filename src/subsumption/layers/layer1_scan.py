"""
src/subsumption/layers/layer1_scan.py

Layer 1: Scan Around — zigzag (boustrophedon) floor-coverage patrol.

--- WHY ZIGZAG ---
The base has no wheel encoders, IMU or odometry, so map-based coverage
(spirals, cell decomposition, SLAM) is out: we cannot know where we are.
What we CAN do without localization is react to events:

    lane 1 -->  drive straight until the ultrasonic sees the wall
    turn        timed ~90 deg pivot (calibrated seconds, not degrees)
    shift       short forward hop of ~one robot width (the lane spacing)
    turn        second timed ~90 deg pivot, same direction
    lane 2 <--  drive back across the floor ... and alternate forever

Alternating the pivot side (left at one wall, right at the other) is what
makes the path sweep ACROSS the room instead of going back over the same
strip. A lane timeout covers open areas with no wall to bounce off.

--- STATE MACHINE ---
    DRIVE ──(wall < TURN_AT_CM, or lane timeout)──> TURN1   (or BACKOFF first
    BACKOFF ──(timed)──> TURN1                               if we got close)
    TURN1 ──(timed ~90 deg)──> SHIFT
    SHIFT ──(timed, or wall ahead)──> TURN2
    TURN2 ──(timed ~90 deg)──> DRIVE   [pivot side flips for the next wall]

Rule 3 note: the phase/timer state is INTERNAL to this layer and resets
whenever the layer deactivates (a target appeared -> layers 2/3 take over and
drive the robot somewhere else, so any remembered zigzag phase is meaningless
afterwards). No global state, no cross-layer knowledge.

Interplay with Layer 5 (emergency stop): this layer turns away at TURN_AT_CM
(~35 cm) while has_obstacle() only trips inside SensorHub.EMERGENCY_STOP_CM
(~10 cm), so the emergency halt is a backstop, not part of normal scanning.
Known limitation: if something jumps in closer than 10 cm, Layer 5 wins and
holds the robot stopped until the obstacle is removed (it never reverses).

--- CALIBRATE ON THE PI (in this order) ---
    TURN_90_S      time a pivot at TURN_SPEED to sweep ~90 deg. Too small ->
                   lanes not parallel, path drifts diagonally. Too big ->
                   robot heads back the way it came.
    SHIFT_S        forward time covering ~one camera-footprint width. Too
                   small -> overlapping lanes (slow but safe); too big ->
                   unscanned gaps between lanes.
    MAX_LANE_S     longest straight run before turning anyway (bounds the
                   patrol area when no wall is in range).
"""
from __future__ import annotations

import enum
import time
from typing import Any, Optional

from src.subsumption.layers.base_layer import BaseLayer
from src.subsumption.arbitrator import ActionCommand

# Motion fractions (scaled by MotionExecutor / MotionCalibration duty).
FORWARD_SPEED = 0.2   # lane cruising speed
TURN_SPEED    = 0.2 # pivot speed (below this the base tends to stall)
# Backing off reuses the live lane speed; a constant here would override
# every speed calibration.

# Ultrasonic thresholds (cm).
TURN_AT_CM    = 35.0  # end the lane and start the zigzag turn
BACKOFF_AT_CM = 20.0  # we noticed the wall late -> reverse first for clearance

# Pivot-side override. Sides are compared saturated at PIVOT_ROOM_CM so a
# harmless far wall can't outvote a side reading nothing at all; the margin
# stops near-equal readings from cancelling the zigzag's alternation.
PIVOT_ROOM_CM   = 40.0
PIVOT_TIGHT_CM  = 25.0
PIVOT_MARGIN_CM = 5.0

# Timed phases (seconds) — calibrate on the Pi, see module docstring.
TURN_90_S  = 0.9
SHIFT_S    = 1.2
BACKOFF_S  = 0.45
MAX_LANE_S = 12.0


def _side_room_cm(sensors: Any, getter: str) -> float:
    """Free space on one side, saturated at PIVOT_ROOM_CM. No echo (None) and
    a missing getter both mean 'nothing in range', i.e. fully open."""
    fn = getattr(sensors, getter, None)
    value = fn() if fn is not None else None
    return PIVOT_ROOM_CM if value is None else min(value, PIVOT_ROOM_CM)


class _Phase(enum.Enum):
    DRIVE   = enum.auto()
    BACKOFF = enum.auto()
    TURN1   = enum.auto()
    SHIFT   = enum.auto()
    TURN2   = enum.auto()


class ScanAroundLayer(BaseLayer):
    """
    Layer 1: Scan Around
    Priority: 1 (Very Low)
    Behavior: Zigzag floor-coverage patrol searching for targets with the
              camera, using the ultrasonic to turn away from walls/obstacles.
              Activates only while no target is detected.
    """

    def __init__(self, forward_speed: float = FORWARD_SPEED,
                 turn_speed: float = TURN_SPEED,
                 turn_90_s: float = TURN_90_S,
                 shift_s: float = SHIFT_S,
                 max_lane_s: float = MAX_LANE_S):
        super().__init__(layer_id=1)
        self.forward_speed = FORWARD_SPEED
        self.turn_speed = TURN_SPEED
        self.turn_90_s = TURN_90_S
        self.shift_s = SHIFT_S
        self.max_lane_s = MAX_LANE_S
        self.set_speeds(forward_speed, turn_speed)
        self.set_timing(turn_90_s, shift_s, max_lane_s)
        self._phase: Optional[_Phase] = None   # None = pattern not started
        self._phase_started: float = 0.0
        self._lane_started: float = 0.0
        self._turn_left: bool = True           # pivot side; alternates per wall
        self._pivot_note: str = "L-- R--"      # side readings behind the last choice
        self._suppressed_since: Optional[float] = None

    # ── Calibration knobs (the Pi tools drive these, see module docstring) ─

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
        # Clamped to the 0..1 motion-fraction contract MotionExecutor expects.
        if forward_speed is not None:
            self.forward_speed = max(0.0, min(1.0, float(forward_speed)))
        if turn_speed is not None:
            self.turn_speed = max(0.0, min(1.0, float(turn_speed)))

    def timing_summary(self) -> str:
        return (f"turn_90={self.turn_90_s:.2f}s shift={self.shift_s:.2f}s "
                f"max_lane={self.max_lane_s:.1f}s fwd={self.forward_speed:.2f} "
                f"turn={self.turn_speed:.2f}")

    # ── Subsumption API ───────────────────────────────────────────────────

    def notify_arbitration(self, won: bool) -> None:
        if not won and self._suppressed_since is None:
            self._suppressed_since = time.monotonic()

    def evaluate(self, sensors: Any) -> ActionCommand:
        # A target exists -> higher layers will handle it; go inactive and
        # forget the zigzag phase (the robot is about to move off-pattern).
        if sensors.get_litter_position() or sensors.get_aerial_trash_position():
            self._phase = None
            self._suppressed_since = None
            return ActionCommand(layer_id=self.layer_id, active=False)

        now = time.monotonic()
        # Every phase is timed open-loop, so time spent suppressed by a higher
        # layer must not count -- otherwise the pattern runs to completion
        # while the robot is being driven somewhere else entirely.
        if self._suppressed_since is not None:
            paused = now - self._suppressed_since
            self._phase_started += paused
            self._lane_started += paused
            self._suppressed_since = None

        dist = self._obstacle_distance_cm(sensors)

        if self._phase is None:
            self._enter(_Phase.DRIVE, now)
            self._lane_started = now

        # ---- phase transitions ------------------------------------------
        if self._phase == _Phase.DRIVE:
            wall_seen = dist is not None and dist <= TURN_AT_CM
            if wall_seen and dist <= BACKOFF_AT_CM:
                self._enter(_Phase.BACKOFF, now)
            elif wall_seen or (now - self._lane_started) >= self.max_lane_s:
                self._turn_left = self._pivot_side(sensors)
                self._enter(_Phase.TURN1, now)

        elif self._phase == _Phase.BACKOFF:
            if self._elapsed(now) >= BACKOFF_S:
                self._turn_left = self._pivot_side(sensors)
                self._enter(_Phase.TURN1, now)

        elif self._phase == _Phase.TURN1:
            if self._elapsed(now) >= self.turn_90_s:
                self._enter(_Phase.SHIFT, now)

        elif self._phase == _Phase.SHIFT:
            # Corner case: wall ahead during the shift -> skip straight to
            # the second pivot instead of driving into it.
            wall_seen = dist is not None and dist <= TURN_AT_CM
            if wall_seen or self._elapsed(now) >= self.shift_s:
                self._enter(_Phase.TURN2, now)

        elif self._phase == _Phase.TURN2:
            if self._elapsed(now) >= self.turn_90_s:
                self._turn_left = not self._turn_left  # alternate -> zigzag
                self._enter(_Phase.DRIVE, now)
                self._lane_started = now

        # ---- phase outputs ----------------------------------------------
        motion, label = self._motion_for_phase()
        dist_txt = f"{dist:.0f}cm" if dist is not None else "--"
        return ActionCommand(
            layer_id=self.layer_id,
            active=True,
            motion_vector=motion,
            arm_action='stow',                 # arm stays stowed while patrolling
            message=f"SCAN {label} (wall {dist_txt})",
        )

    # ── Internals ─────────────────────────────────────────────────────────

    def _motion_for_phase(self):
        """(v_x, v_y, v_theta) for the current phase. v_theta > 0 = CCW/left."""
        turn = self.turn_speed if self._turn_left else -self.turn_speed
        if self._phase == _Phase.DRIVE:
            return (self.forward_speed, 0, 0), "lane"
        if self._phase == _Phase.BACKOFF:
            return (-self.forward_speed, 0, 0), "backing off wall"
        side = 'left' if self._turn_left else 'right'
        if self._phase == _Phase.TURN1:
            return (0, 0, turn), f"turn 1 ({side}, v_theta {turn:+.2f}, {self._pivot_note})"
        if self._phase == _Phase.SHIFT:
            return (self.forward_speed, 0, 0), "shifting lane"
        if self._phase == _Phase.TURN2:
            return (0, 0, turn), f"turn 2 ({side}, v_theta {turn:+.2f}, {self._pivot_note})"
        return (0, 0, 0), "idle"

    def _enter(self, phase: _Phase, now: float) -> None:
        self._phase = phase
        self._phase_started = now

    def _elapsed(self, now: float) -> float:
        return now - self._phase_started

    def _pivot_side(self, sensors: Any) -> bool:
        """Which way to swing this lane change. Alternating is what makes the
        pattern a zigzag, so keep it -- but never swing into the tighter side
        when the other one is clearly roomier, or a corner just traps the
        robot pivoting back and forth into the same wall."""
        left = _side_room_cm(sensors, "get_obstacle_distance_front_left_cm")
        right = _side_room_cm(sensors, "get_obstacle_distance_front_right_cm")
        self._pivot_note = f"L{left:.0f} R{right:.0f}"
        intended, other = (left, right) if self._turn_left else (right, left)
        # Only a genuinely tight intended side justifies breaking alternation,
        # and then any clearly roomier alternative wins -- requiring the old
        # wide margin left the robot pivoting into the nearer of two close walls.
        if intended < PIVOT_TIGHT_CM and other >= intended + PIVOT_MARGIN_CM:
            return not self._turn_left
        return self._turn_left

    @staticmethod
    def _obstacle_distance_cm(sensors: Any) -> Optional[float]:
        """Nearest of the three forward sensors. Reading the front one alone
        left the diagonals invisible here, so a wall off to one side was
        never seen at TURN_AT_CM and the pattern only ever gave way to the
        layer 5 escape. Missing getters tolerate older sensor objects."""
        readings = []
        for name in ("get_obstacle_distance_cm",
                     "get_obstacle_distance_front_left_cm",
                     "get_obstacle_distance_front_right_cm"):
            getter = getattr(sensors, name, None)
            value = getter() if getter is not None else None
            if value is not None:
                readings.append(value)
        return min(readings) if readings else None
