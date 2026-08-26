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
Something ahead is not automatically a wall. The robot DODGES first and only
treats the blockage as a wall when the dodge fails to get past it, so a bin in
the middle of the floor no longer ends the lane and costs a whole strip.

    DRIVE ──(front < TURN_AT_CM)──> DODGE_TURN   (or BACKOFF first if close)
    DRIVE ──(lane timeout)──> TURN1
    BACKOFF ──(timed)──> DODGE_TURN
    DODGE_TURN ──(timed ~90 deg toward the open side)──> DODGE_PASS
    DODGE_PASS ──(the blocked side reads clear)──> DODGE_BACK   [obstacle]
    DODGE_PASS ──(DODGE_MAX_S, or blocked again)──> TURN2       [wall]
    DODGE_BACK ──(timed ~90 deg back)──> DRIVE   [same lane, keeps going]
    TURN1 ──(timed ~90 deg)──> SHIFT
    SHIFT ──(timed, or wall ahead)──> TURN2
    TURN2 ──(timed ~90 deg)──> DRIVE   [pivot side flips for the next wall]

The wall path reuses the dodge itself as the first half of the lane change:
DODGE_TURN is the first pivot and DODGE_PASS runs along the wall exactly as
SHIFT did, so DODGE_TURN + DODGE_PASS + TURN2 is the same ~180 deg
turn-and-offset the zigzag always did. Failing to dodge costs nothing.

Only the FRONT sensor ends a lane. A close diagonal means a wall alongside,
not a wall in the way: it steers the lane away instead (see _lane_bias). Live
log that forced this -- the robot dodged with `front 43cm`, a clear road
ahead, because a side wall read 17cm; each failed dodge is a 180, so it
about-faced back and forth in a corridor and never drove out of it.

Every phase change passes through SETTLE first: the wheels reverse direction
between almost any two phases (lane -> pivot flips one wheel, pivot -> lane
flips the other), and instant flips spike current, stall the driver and brown
out the rail (see docs/hardware_safety_patterns.md rule 7).

Before every new lane (at boot, and again each time TURN2 finishes), the
robot does a full look-around: eight 45 deg pivot steps with a dwell pause
after each one, so the camera gets a still frame at every heading around the
robot instead of relying on the drive-by pass to catch a target. This is a
pure in-place pivot, so it needs no obstacle check of its own -- it never
translates, so it cannot drive into anything while turning.

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
import random
import time
from typing import Any, Optional

from src.subsumption.layers.base_layer import BaseLayer
from src.subsumption.arbitrator import ActionCommand

# Motion fractions (scaled by MotionExecutor / MotionCalibration duty).
FORWARD_SPEED = 0.20   # lane cruising speed
TURN_SPEED    = 0.30 # pivot speed (below this the base tends to stall)
# Backing off reuses the live lane speed; a constant here would override
# every speed calibration.

# Ultrasonic thresholds (cm).
TURN_AT_CM    = 20.0  # front sensor: end the lane and start the zigzag turn
BACKOFF_AT_CM = 15.0  # we noticed the wall late -> reverse first for clearance
BACKOFF_REAR_MIN_CM = 20.0  # abort the reverse if the rear closes to this

# Diagonals steer the lane, they never end it. Full gain (at zero clearance)
# is NUDGE_GAIN of the pivot speed, so the correction is always gentler than
# a deliberate turn and two opposing walls cancel to a straight line.
DIAGONAL_NUDGE_CM = 30.0
NUDGE_GAIN        = 0.35

# Pivot-side override. Sides are compared saturated at PIVOT_ROOM_CM so a
# harmless far wall can't outvote a side reading nothing at all; the margin
# stops near-equal readings from cancelling the zigzag's alternation.
PIVOT_ROOM_CM   = 40.0
PIVOT_TIGHT_CM  = 25.0
PIVOT_MARGIN_CM = 5.0

# Dodge: what separates an obstacle from a wall. After pivoting away the
# robot watches the diagonal now pointing at the blockage; a side that goes
# clear means it has been passed, a side still blocked means a wall.
DODGE_CLEAR_CM = 30.0  # that diagonal must open up to at least this
DODGE_MIN_S    = 0.5   # ignore a clear side before this: a 45 deg beam may
                       # never have caught the obstacle in the first place
DODGE_MAX_S    = 2.0   # still blocked after this -> wall, do the lane change

# Timed phases (seconds) — calibrate on the Pi, see module docstring.
TURN_90_S  = 0.9
SHIFT_S    = 1.2
BACKOFF_S  = 0.45
MAX_LANE_S = 30.0
SETTLE_S   = 0.15   # wheels stop between phases before they flip direction

# Every timed phase is re-rolled +/- this fraction. Fixed durations make a
# fixed path: with no odometry the same turn and the same lane length walk the
# robot back over the strip it just did, so it circles one patch forever.
# Set to 0.0 for a deterministic pattern (the unit tests do).
TIMING_JITTER = 0.2

# Full look-around before each new lane: eight 45 deg steps, not jittered, so
# the eight steps still sum to a clean 360 and the sweep does not drift.
LOOKAROUND_STEPS    = 8
LOOKAROUND_STEP_DEG = 360.0 / LOOKAROUND_STEPS
LOOKAROUND_DWELL_S  = 0.6   # hold still here so a frame can settle and be inferred


def _jitter(seconds: float) -> float:
    if TIMING_JITTER <= 0.0:
        return seconds
    return seconds * random.uniform(1.0 - TIMING_JITTER, 1.0 + TIMING_JITTER)


def _fmt(value) -> str:
    return "--" if value is None else f"{value:.0f}"


def _side_room_cm(sensors: Any, getter: str) -> float:
    """Free space on one side, saturated at PIVOT_ROOM_CM. No echo (None) and
    a missing getter both mean 'nothing in range', i.e. fully open."""
    fn = getattr(sensors, getter, None)
    value = fn() if fn is not None else None
    return PIVOT_ROOM_CM if value is None else min(value, PIVOT_ROOM_CM)


class _Phase(enum.Enum):
    SETTLE          = enum.auto()   # brief halt between phases (direction flips)
    LOOKAROUND_TURN = enum.auto()   # one 45 deg step of the pre-lane look-around
    LOOKAROUND_DWELL = enum.auto()  # holds still after a step so a frame settles
    DRIVE      = enum.auto()
    BACKOFF    = enum.auto()
    DODGE_TURN = enum.auto()   # pivot away from whatever is ahead
    DODGE_PASS = enum.auto()   # drive past it, watching the side it is on
    DODGE_BACK = enum.auto()   # pivot back onto the lane heading
    TURN1      = enum.auto()
    SHIFT      = enum.auto()
    TURN2      = enum.auto()


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
        self._lane_limit_s: float = self.max_lane_s   # this lane's jittered cap
        self._turn_s: float = self.turn_90_s          # this pivot's jittered time
        self._turn_left: bool = True           # pivot side; alternates per wall
        self._dodge_left: bool = True          # which way the current dodge went
        self._lookaround_step: int = 0         # completed steps of the pre-lane sweep
        # True only for a GENUINE fresh start (boot, operator reset, or a
        # completed lane change) -- NOT for a target that merely flickered out
        # of view for a tick and handed control straight back. Without this,
        # a flickering detection retriggers the full 8-step sweep on every
        # single dropped frame, fighting the very approach it just yielded to.
        self._needs_lookaround: bool = True
        self._next_phase: _Phase = _Phase.DRIVE  # what the settle is settling for
        self._pivot_note: str = "L-- R--"      # side readings behind the last choice
        self._suppressed_since: Optional[float] = None
        # Only stand down for a target when a layer above is there to chase
        # it. SCAN-only mode has no Layer 2, so yielding parked the robot on
        # Layer 0 IDLE for as long as the camera could see a can.
        self.yield_to_targets: bool = True

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

    def reset(self) -> None:
        self._phase = None
        self._needs_lookaround = True  # operator/mode change: re-scan on resume
        self._suppressed_since = None

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
        # Aerial trash is deliberately NOT a reason: Layer 4 is not in any
        # running stack, so nothing would take over.
        if self.yield_to_targets and sensors.get_litter_position():
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

        front, left, right = self._clearances(sensors)
        wall_ahead = front is not None and front <= TURN_AT_CM

        if self._phase is None:
            if self._needs_lookaround:
                self._lookaround_step = 0
                self._enter(_Phase.LOOKAROUND_TURN, now)
            else:
                # Just a flickered detection handing control back -- resume
                # driving plainly, no re-scan, no lane restart.
                self._enter(_Phase.DRIVE, now)
                self._start_lane(now)

        # ---- phase transitions ------------------------------------------
        if self._phase == _Phase.SETTLE:
            if self._elapsed(now) >= SETTLE_S:
                self._enter(self._next_phase, now)

        elif self._phase == _Phase.LOOKAROUND_TURN:
            if self._elapsed(now) >= self._lookaround_step_s():
                self._enter(_Phase.LOOKAROUND_DWELL, now)

        elif self._phase == _Phase.LOOKAROUND_DWELL:
            if self._elapsed(now) >= LOOKAROUND_DWELL_S:
                self._lookaround_step += 1
                if self._lookaround_step >= LOOKAROUND_STEPS:
                    self._needs_lookaround = False
                    self._start_lane(now)
                    self._enter(_Phase.DRIVE, now)
                else:
                    self._enter(_Phase.LOOKAROUND_TURN, now)

        elif self._phase == _Phase.DRIVE:
            if wall_ahead and front <= BACKOFF_AT_CM:
                self._settle(_Phase.BACKOFF, now)
            elif wall_ahead:
                self._start_dodge(sensors, now)
            elif (now - self._lane_started) >= self._lane_limit_s:
                self._turn_left = self._pivot_side(sensors)
                self._settle(_Phase.TURN1, now)

        elif self._phase == _Phase.BACKOFF:
            # Cut the reverse short if something is behind us: this is the only
            # phase in the pattern that moves backwards, so it is the only one
            # the rear sensor has any say over.
            rear = _side_room_cm(sensors, "get_obstacle_distance_back_cm")
            if self._elapsed(now) >= BACKOFF_S or rear < BACKOFF_REAR_MIN_CM:
                self._start_dodge(sensors, now)

        elif self._phase == _Phase.DODGE_TURN:
            if self._elapsed(now) >= self.turn_90_s:
                self._settle(_Phase.DODGE_PASS, now)

        elif self._phase == _Phase.DODGE_PASS:
            elapsed = self._elapsed(now)
            if elapsed >= DODGE_MAX_S or wall_ahead:
                # Cannot get round it: treat it as a wall. Geometrically the
                # dodge already IS the first pivot and the sideways hop, so the
                # lane change only has its second pivot left to run.
                self._turn_left = self._dodge_left
                self._settle(_Phase.TURN2, now)
            elif elapsed >= DODGE_MIN_S and self._dodge_side_clear(sensors):
                self._settle(_Phase.DODGE_BACK, now)

        elif self._phase == _Phase.DODGE_BACK:
            if self._elapsed(now) >= self.turn_90_s:
                self._settle(_Phase.DRIVE, now)

        elif self._phase == _Phase.TURN1:
            if self._elapsed(now) >= self._turn_s:
                self._settle(_Phase.SHIFT, now)

        elif self._phase == _Phase.SHIFT:
            # Corner case: wall ahead during the shift -> skip straight to
            # the second pivot instead of driving into it.
            if wall_ahead or self._elapsed(now) >= self.shift_s:
                self._settle(_Phase.TURN2, now)

        elif self._phase == _Phase.TURN2:
            if self._elapsed(now) >= self._turn_s:
                self._turn_left = not self._turn_left  # alternate -> zigzag
                self._lookaround_step = 0
                self._enter(_Phase.LOOKAROUND_TURN, now)

        return self._output(front, left, right)

    # ── Internals ─────────────────────────────────────────────────────────

    def _output(self, front, left, right) -> ActionCommand:
        motion, label = self._motion_for_phase(left, right)
        front_txt = f"{front:.0f}cm" if front is not None else "--"
        return ActionCommand(
            layer_id=self.layer_id,
            active=True,
            motion_vector=motion,
            arm_action='stow',                 # arm stays stowed while patrolling
            message=f"SCAN {label} (front {front_txt})",
        )

    def _lane_bias(self, left, right) -> float:
        """Steer the lane away from a wall alongside, hardest when closest.
        Opposing walls cancel, which is what centres the robot in a corridor
        instead of bouncing it off one side."""
        bias = 0.0
        if left is not None and left < DIAGONAL_NUDGE_CM:
            bias -= NUDGE_GAIN * (1.0 - left / DIAGONAL_NUDGE_CM)
        if right is not None and right < DIAGONAL_NUDGE_CM:
            bias += NUDGE_GAIN * (1.0 - right / DIAGONAL_NUDGE_CM)
        return max(-1.0, min(1.0, bias)) * self.turn_speed

    def _motion_for_phase(self, left=None, right=None):
        """(v_x, v_y, v_theta) for the current phase. v_theta > 0 = CCW/left."""
        turn = self.turn_speed if self._turn_left else -self.turn_speed
        if self._phase == _Phase.SETTLE:
            return (0, 0, 0), f"settling before {self._next_phase.name.lower()}"
        if self._phase == _Phase.LOOKAROUND_TURN:
            return (0, 0, turn), f"look-around step {self._lookaround_step + 1}/{LOOKAROUND_STEPS}"
        if self._phase == _Phase.LOOKAROUND_DWELL:
            return (0, 0, 0), f"look-around dwell {self._lookaround_step + 1}/{LOOKAROUND_STEPS}"
        if self._phase == _Phase.DRIVE:
            bias = self._lane_bias(left, right)
            if bias == 0.0:
                return (self.forward_speed, 0, 0), "lane"
            return ((self.forward_speed, 0, bias),
                    f"lane (nudge {bias:+.2f}, L{_fmt(left)} R{_fmt(right)})")
        if self._phase == _Phase.BACKOFF:
            return (-self.forward_speed, 0, 0), "backing off wall"
        dodge = self.turn_speed if self._dodge_left else -self.turn_speed
        dodge_side = 'left' if self._dodge_left else 'right'
        if self._phase == _Phase.DODGE_TURN:
            return (0, 0, dodge), f"dodging {dodge_side} ({self._pivot_note})"
        if self._phase == _Phase.DODGE_PASS:
            return (self.forward_speed, 0, 0), f"passing obstacle on the {'right' if self._dodge_left else 'left'}"
        if self._phase == _Phase.DODGE_BACK:
            return (0, 0, -dodge), "back onto the lane"
        side = 'left' if self._turn_left else 'right'
        if self._phase == _Phase.TURN1:
            return (0, 0, turn), f"turn 1 ({side}, v_theta {turn:+.2f}, {self._pivot_note})"
        if self._phase == _Phase.SHIFT:
            return (self.forward_speed, 0, 0), "shifting lane"
        if self._phase == _Phase.TURN2:
            return (0, 0, turn), f"turn 2 ({side}, v_theta {turn:+.2f}, {self._pivot_note})"
        return (0, 0, 0), "idle"

    def _settle(self, nxt: _Phase, now: float) -> None:
        """Halt briefly before the wheels flip direction, then run `nxt`."""
        self._next_phase = nxt
        self._phase = _Phase.SETTLE
        self._phase_started = now

    def _enter(self, phase: _Phase, now: float) -> None:
        self._phase = phase
        self._phase_started = now
        if phase in (_Phase.TURN1, _Phase.TURN2):
            self._turn_s = _jitter(self.turn_90_s)

    def _start_lane(self, now: float) -> None:
        self._lane_started = now
        self._lane_limit_s = _jitter(self.max_lane_s)

    def _lookaround_step_s(self) -> float:
        """Pivot time for one 45 deg look-around step, scaled off the same
        calibrated TURN_90_S the zigzag pivots use (not jittered, see the
        module docstring)."""
        return self.turn_90_s * (LOOKAROUND_STEP_DEG / 90.0)

    def _elapsed(self, now: float) -> float:
        return now - self._phase_started

    def _start_dodge(self, sensors: Any, now: float) -> None:
        """Pivot toward whichever diagonal has more room. Unlike a lane change
        this never alternates for its own sake: dodging into the tighter side
        is how the robot used to wedge itself in a corner."""
        left = _side_room_cm(sensors, "get_obstacle_distance_front_left_cm")
        right = _side_room_cm(sensors, "get_obstacle_distance_front_right_cm")
        self._pivot_note = f"L{left:.0f} R{right:.0f}"
        self._dodge_left = left >= right
        self._settle(_Phase.DODGE_TURN, now)

    def _dodge_side_clear(self, sensors: Any) -> bool:
        """After pivoting away, the blockage sits on the opposite diagonal.
        That side opening up means the robot is past it."""
        getter = ("get_obstacle_distance_front_right_cm" if self._dodge_left
                  else "get_obstacle_distance_front_left_cm")
        return _side_room_cm(sensors, getter) >= DODGE_CLEAR_CM

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
    def _clearances(sensors: Any):
        """(front, front_left, front_right) in cm; None = nothing in range.
        Missing getters tolerate older sensor objects."""
        out = []
        for name in ("get_obstacle_distance_cm",
                     "get_obstacle_distance_front_left_cm",
                     "get_obstacle_distance_front_right_cm"):
            getter = getattr(sensors, name, None)
            out.append(getter() if getter is not None else None)
        return tuple(out)

