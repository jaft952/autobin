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

# Timed phases (seconds) — calibrate on the Pi, see module docstring.
TURN_90_S  = 0.9
SHIFT_S    = 1.2
BACKOFF_S  = 0.45
MAX_LANE_S = 12.0


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

    def evaluate(self, sensors: Any) -> ActionCommand:
        # A target exists -> higher layers will handle it; go inactive and
        # forget the zigzag phase (the robot is about to move off-pattern).
        if sensors.get_litter_position() or sensors.get_aerial_trash_position():
            self._phase = None
            return ActionCommand(layer_id=self.layer_id, active=False)

        now = time.monotonic()
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
                self._enter(_Phase.TURN1, now)

        elif self._phase == _Phase.BACKOFF:
            if self._elapsed(now) >= BACKOFF_S:
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
        if self._phase == _Phase.TURN1:
            return (0, 0, turn), f"turn 1 ({'left' if self._turn_left else 'right'})"
        if self._phase == _Phase.SHIFT:
            return (self.forward_speed, 0, 0), "shifting lane"
        if self._phase == _Phase.TURN2:
            return (0, 0, turn), f"turn 2 ({'left' if self._turn_left else 'right'})"
        return (0, 0, 0), "idle"

    def _enter(self, phase: _Phase, now: float) -> None:
        self._phase = phase
        self._phase_started = now

    def _elapsed(self, now: float) -> float:
        return now - self._phase_started

    @staticmethod
    def _obstacle_distance_cm(sensors: Any) -> Optional[float]:
        """Tolerate sensor objects predating get_obstacle_distance_cm()."""
        getter = getattr(sensors, "get_obstacle_distance_cm", None)
        return getter() if getter is not None else None
