"""
tests/test_scan_logic.py

Pure-logic tests for the zigzag (boustrophedon) floor-scan algorithm.
NO hardware needed — runs on Windows or the Pi:

    python tests/test_scan_logic.py      # plain runner with PASS/FAIL summary
    pytest tests/test_scan_logic.py      # also works

What it does: replaces time.monotonic with a fake clock so the timed phases
(TURN_90_S, SHIFT_S, ...) can be fast-forwarded instantly, and feeds the layer
a fake sensors object whose ultrasonic distance we script. Covers:

    - full zigzag cycle: DRIVE -> TURN1 -> SHIFT -> TURN2 -> DRIVE,
      and the pivot side alternating at the next wall
    - late wall (< BACKOFF_AT_CM) -> reverses before pivoting
    - open floor (no wall) -> lane timeout still turns
    - wall appearing mid-SHIFT (corner) -> skips straight to TURN2
    - target detected -> layer yields and restarts the pattern afterwards
    - MotionExecutor: vector -> wheel mixing, sign convention, renormalize, stop
    - arbitration: scan beats idle; emergency (< 10 cm) beats scan

For the on-robot integration test (real ultrasonic + wheels) use
tests/test_floor_scan.py instead.
"""
import sys
from contextlib import contextmanager
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import src.subsumption.layers.layer1_scan as scan_mod
from src.subsumption.layers.layer1_scan import (
    ScanAroundLayer,
    FORWARD_SPEED, TURN_SPEED,
    TURN_AT_CM, BACKOFF_AT_CM,
    TURN_90_S, SHIFT_S, BACKOFF_S, MAX_LANE_S,
)
from src.subsumption.arbitrator import Arbitrator, ActionCommand
from src.subsumption.layers.layer0_idle import SystemIdleLayer
import src.subsumption.layers.layer5_emergency as emergency_mod
from src.subsumption.layers.layer5_emergency import EmergencyStopLayer, EMERGENCY_TURN_SPEED
from src.subsumption.motion_executor import MotionExecutor
from src.hardware.sensors.sensor_hub import SensorHub


# ── Test doubles ──────────────────────────────────────────────────────────

class FakeClock:
    """Stands in for time.monotonic so timed phases can be fast-forwarded."""

    def __init__(self, start: float = 1000.0):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def tick(self, seconds: float) -> None:
        self.t += seconds


@contextmanager
def fake_clock():
    """Patch time.monotonic for the scan module, restore on exit."""
    clock = FakeClock()
    real = scan_mod.time.monotonic
    scan_mod.time.monotonic = clock
    try:
        yield clock
    finally:
        scan_mod.time.monotonic = real


class FakeSensors:
    """Scriptable stand-in for the SensorHub, seen from a layer's viewpoint."""

    def __init__(self):
        self.dist = None    # ultrasonic reading (cm), None = nothing in range
        self.litter = None  # (x, y) or None

    def get_litter_position(self):
        return self.litter

    def get_aerial_trash_position(self):
        return None

    def get_obstacle_distance_cm(self):
        return self.dist


class FakeUltrasonic:
    """Minimal UltrasonicSensor stand-in for driving a real SensorHub."""

    def __init__(self):
        self.dist = None

    def update(self):
        pass

    def get_distance_cm(self):
        return self.dist

    def close(self):
        pass


class CaptureActuator:
    """Records what the MotionExecutor would send to the wheels."""

    def __init__(self):
        self.last = None
        self.stopped = False
        self.braked = False

    def apply(self, cmd):
        self.last = (round(cmd.left_speed, 1), round(cmd.right_speed, 1), cmd.trim_set)
        self.stopped = False
        self.braked = False

    def stop(self):
        self.stopped = True
        self.braked = False

    def brake(self):
        self.braked = True
        self.stopped = False

    def close(self):
        pass


# ── Scan layer: the zigzag state machine ─────────────────────────────────

def test_full_zigzag_cycle():
    """One complete lane change, then the pivot side alternates."""
    with fake_clock() as clock:
        layer = ScanAroundLayer()
        sensors = FakeSensors()

        # Open floor -> cruising down the lane.
        cmd = layer.evaluate(sensors)
        assert cmd.active
        assert cmd.motion_vector == (FORWARD_SPEED, 0, 0), cmd.message
        assert cmd.arm_action == 'stow'

        # Wall enters the turn band (but not the backoff band) -> pivot 1,
        # first pivot of the pattern is LEFT (positive v_theta = CCW).
        sensors.dist = (TURN_AT_CM + BACKOFF_AT_CM) / 2
        cmd = layer.evaluate(sensors)
        assert cmd.motion_vector == (0, 0, TURN_SPEED), cmd.message

        # Pivot keeps going until TURN_90_S has elapsed...
        clock.tick(TURN_90_S * 0.5)
        cmd = layer.evaluate(sensors)
        assert cmd.motion_vector == (0, 0, TURN_SPEED), cmd.message

        # ...then shifts one lane-width forward (wall now beside us).
        clock.tick(TURN_90_S * 0.5 + 0.01)
        sensors.dist = None
        cmd = layer.evaluate(sensors)
        assert cmd.motion_vector == (FORWARD_SPEED, 0, 0), cmd.message
        assert "shift" in cmd.message.lower()

        # Shift done -> pivot 2, SAME side, completing the 180.
        clock.tick(SHIFT_S + 0.01)
        cmd = layer.evaluate(sensors)
        assert cmd.motion_vector == (0, 0, TURN_SPEED), cmd.message

        # Pivot 2 done -> back to cruising the return lane.
        clock.tick(TURN_90_S + 0.01)
        cmd = layer.evaluate(sensors)
        assert cmd.motion_vector == (FORWARD_SPEED, 0, 0), cmd.message

        # Next wall -> pivot side has ALTERNATED to the right (that's the
        # "zig" vs "zag" — without this the robot retraces the same strip).
        sensors.dist = TURN_AT_CM - 1
        cmd = layer.evaluate(sensors)
        assert cmd.motion_vector == (0, 0, -TURN_SPEED), cmd.message
    print("PASS full zigzag cycle + side alternation")


def test_backoff_when_wall_seen_late():
    """Wall closer than BACKOFF_AT_CM -> reverse first to make pivot room."""
    with fake_clock() as clock:
        layer = ScanAroundLayer()
        sensors = FakeSensors()
        layer.evaluate(sensors)  # enter DRIVE

        sensors.dist = BACKOFF_AT_CM - 5
        cmd = layer.evaluate(sensors)
        assert cmd.motion_vector == (-FORWARD_SPEED, 0, 0), cmd.message

        # Backoff is timed; afterwards the normal pivot starts.
        clock.tick(BACKOFF_S + 0.01)
        cmd = layer.evaluate(sensors)
        assert cmd.motion_vector == (0, 0, TURN_SPEED), cmd.message
    print("PASS backoff before pivot when wall is close")


def test_lane_timeout_without_wall():
    """Open area, ultrasonic never fires -> MAX_LANE_S bounds the lane."""
    with fake_clock() as clock:
        layer = ScanAroundLayer()
        sensors = FakeSensors()  # dist stays None
        cmd = layer.evaluate(sensors)
        assert cmd.motion_vector == (FORWARD_SPEED, 0, 0)

        # Just before the timeout: still driving.
        clock.tick(MAX_LANE_S - 0.1)
        cmd = layer.evaluate(sensors)
        assert cmd.motion_vector == (FORWARD_SPEED, 0, 0), cmd.message

        # Past it: turns anyway.
        clock.tick(0.2)
        cmd = layer.evaluate(sensors)
        assert cmd.motion_vector == (0, 0, TURN_SPEED), cmd.message
    print("PASS lane timeout turns without a wall")


def test_corner_wall_during_shift():
    """Wall ahead during the SHIFT hop (corner) -> skip straight to pivot 2."""
    with fake_clock() as clock:
        layer = ScanAroundLayer()
        sensors = FakeSensors()
        layer.evaluate(sensors)                    # DRIVE
        sensors.dist = TURN_AT_CM - 1
        layer.evaluate(sensors)                    # TURN1
        clock.tick(TURN_90_S + 0.01)
        sensors.dist = None
        cmd = layer.evaluate(sensors)              # SHIFT
        assert "shift" in cmd.message.lower(), cmd.message

        # Corner: wall reappears ahead well before SHIFT_S is up.
        clock.tick(SHIFT_S * 0.2)
        sensors.dist = TURN_AT_CM - 1
        cmd = layer.evaluate(sensors)
        assert cmd.motion_vector == (0, 0, TURN_SPEED), cmd.message
        assert "turn 2" in cmd.message.lower(), cmd.message
    print("PASS corner: wall during shift skips to pivot 2")


def test_yields_to_target_and_restarts():
    """Litter detected -> inactive (layers 2/3 take over); pattern restarts
    from a fresh lane when the target is gone, mid-pivot state is forgotten."""
    with fake_clock() as clock:
        layer = ScanAroundLayer()
        sensors = FakeSensors()
        layer.evaluate(sensors)                    # DRIVE
        sensors.dist = TURN_AT_CM - 1
        cmd = layer.evaluate(sensors)              # mid TURN1
        assert cmd.motion_vector[2] != 0

        sensors.litter = (0.4, 0.6)
        cmd = layer.evaluate(sensors)
        assert not cmd.active

        # Target gone (collected / lost) -> fresh DRIVE, not a resumed pivot:
        # the robot moved during approach/grasp, the old phase is meaningless.
        sensors.litter = None
        sensors.dist = None
        cmd = layer.evaluate(sensors)
        assert cmd.active
        assert cmd.motion_vector == (FORWARD_SPEED, 0, 0), cmd.message
    print("PASS yields to target and restarts pattern")


def test_none_distance_is_not_an_obstacle():
    """None (no echo / dev machine) must mean 'no wall info', never 0 cm."""
    with fake_clock():
        layer = ScanAroundLayer()
        sensors = FakeSensors()
        sensors.dist = None
        cmd = layer.evaluate(sensors)
        assert cmd.motion_vector == (FORWARD_SPEED, 0, 0), cmd.message
    print("PASS None distance keeps driving")


# ── MotionExecutor: motion_vector -> wheels ───────────────────────────────

def test_executor_mixing():
    act = CaptureActuator()
    ex = MotionExecutor(actuator=act)

    # Derived from the calibration so re-tuning the base can't stale the test.
    duty = ex.cal.forward_speed

    # Straight lane: both wheels equal, forward trim.
    ex.execute(ActionCommand(1, True, (0.6, 0, 0), None, ""))
    assert act.last == (round(0.6 * duty, 1), round(0.6 * duty, 1), "forward"), act.last

    # Positive v_theta = CCW/left = left wheel back, right wheel forward —
    # must match DifferentialKinematics.turn_left() = (-speed, +speed).
    ex.execute(ActionCommand(1, True, (0, 0, TURN_SPEED), None, ""))
    assert act.last == (round(-TURN_SPEED * duty, 1), round(TURN_SPEED * duty, 1), "turn"), act.last

    # Backoff: both wheels reverse with backward trim.
    ex.execute(ActionCommand(1, True, (-0.5, 0, 0), None, ""))
    assert act.last == (round(-0.5 * duty, 1), round(-0.5 * duty, 1), "backward"), act.last

    # Saturating arc renormalizes (keeps the curve RATIO) instead of clipping.
    ex.execute(ActionCommand(1, True, (1.0, 0, 0.5), None, ""))
    assert act.last == (round(duty / 3, 1), round(duty, 1), "forward"), act.last

    # Zero vector and inactive/idle commands must stop (coast) the base.
    ex.execute(ActionCommand(1, True, (0, 0, 0), None, ""))
    assert act.stopped and not act.braked
    ex.execute(ActionCommand(-1, False, None, None, "Idle"))
    assert act.stopped and not act.braked
    print("PASS executor mixing + sign convention + stop")


def test_executor_brakes_during_grab():
    """A halt (0,0,0) that accompanies a grab must BRAKE (hold position so
    the arm's shaking can't drift the base), not coast."""
    act = CaptureActuator()
    ex = MotionExecutor(actuator=act)

    ex.execute(ActionCommand(3, True, (0, 0, 0), "grab_arc", "",
                             {"pose": [100, 145, 75, 165, 90]}))
    assert act.braked and not act.stopped, "grab halt must brake, not coast"

    ex.execute(ActionCommand(3, True, (0, 0, 0), "grab_ik", "",
                             {"target_m": (0.1, 0.2)}))
    assert act.braked, "IK grab halt must brake too"

    # A plain scan/idle halt still coasts (free to be repositioned).
    ex.execute(ActionCommand(0, True, (0, 0, 0), "stow", ""))
    assert act.stopped and not act.braked

    # Driving again releases the brake (apply overwrites it).
    ex.execute(ActionCommand(1, True, (0.6, 0, 0), None, ""))
    assert not act.braked and not act.stopped
    print("PASS executor brakes during grab, coasts otherwise")


def test_executor_brake_falls_back_to_stop():
    """An actuator with no brake() (e.g. a print stub) must degrade to a
    plain stop instead of crashing."""
    class NoBrakeActuator:
        def __init__(self): self.stopped = False
        def apply(self, cmd): self.stopped = False
        def stop(self): self.stopped = True

    act = NoBrakeActuator()
    ex = MotionExecutor(actuator=act)
    ex.execute(ActionCommand(3, True, (0, 0, 0), "grab_arc", "", {"pose": [0] * 5}))
    assert act.stopped, "brake must fall back to stop when unsupported"
    print("PASS executor brake falls back to stop when unsupported")


# ── Arbitration: scanning combined with the other layers ─────────────────

def _vote(arbitrator, sensors_obj):
    for layer in (SystemIdleLayer(), ScanAroundLayer(), EmergencyStopLayer()):
        arbitrator.submit_command(layer.evaluate(sensors_obj))
    winning = arbitrator.get_winning_action()
    arbitrator.clear()
    return winning


def test_arbitration_with_real_hub():
    with fake_clock():
        ultra = FakeUltrasonic()
        hub = SensorHub(front=ultra, camera=None)
        arb = Arbitrator()

        # Clear floor: scan (layer 1) outvotes idle (layer 0).
        ultra.dist = 120.0
        win = _vote(arb, hub)
        assert win.layer_id == 1, win.message
        assert win.motion_vector == (FORWARD_SPEED, 0, 0), win.message

        # Wall in the turn band: still scan's job (it turns), NOT an emergency.
        ultra.dist = TURN_AT_CM - 5
        win = _vote(arb, hub)
        assert win.layer_id == 1, win.message

        # Inside EMERGENCY_STOP_CM: layer 5 subsumes everything. What it then
        # does (settle/backoff/pivot) is covered by the avoid tests below.
        ultra.dist = SensorHub.EMERGENCY_STOP_CM - 2
        win = _vote(arb, hub)
        assert win.layer_id == 5, win.message
    print("PASS arbitration: scan > idle, emergency > scan")


# ── Layer 5: obstacle avoidance ───────────────────────────────────────────

class FakeDirectionalSensors:
    """SensorHub stand-in with one reading per ultrasonic direction."""

    def __init__(self, front=None, back=None, front_left=None, front_right=None):
        self.front, self.back = front, back
        self.front_left, self.front_right = front_left, front_right
        self.litter = None

    def get_litter_position(self):
        return self.litter

    def get_aerial_trash_position(self):
        return None

    def _near(self, d):
        return d is not None and d < SensorHub.EMERGENCY_STOP_CM

    def has_obstacle(self):
        return any(self._near(d) for d in
                   (self.front, self.back, self.front_left, self.front_right))

    def get_obstacle_distance_cm(self):
        return self.front

    def get_obstacle_distance_back_cm(self):
        return self.back

    def get_obstacle_distance_front_left_cm(self):
        return self.front_left

    def get_obstacle_distance_front_right_cm(self):
        return self.front_right


@contextmanager
def fake_emergency_clock():
    clock = FakeClock()
    real = emergency_mod.time.monotonic
    emergency_mod.time.monotonic = clock
    try:
        yield clock
    finally:
        emergency_mod.time.monotonic = real


def _advance_to_pivot(layer, sensors, clock, limit=40):
    """Step the layer through settle -> backoff -> settle until it pivots."""
    cmd = layer.evaluate(sensors)
    for _ in range(limit):
        if cmd.motion_vector[2] != 0:
            return cmd
        clock.tick(0.1)
        cmd = layer.evaluate(sensors)
    raise AssertionError(f"never reached the pivot: {cmd.message}")


def test_avoid_backs_off_before_pivoting_when_the_rear_is_clear():
    """A rectangular base sweeps its corners wider than its front face, so it
    reverses for turning room before pivoting."""
    with fake_emergency_clock() as clock:
        layer = EmergencyStopLayer()
        sensors = FakeDirectionalSensors(front=8, front_left=90, front_right=90, back=None)

        cmd = layer.evaluate(sensors)
        assert cmd.motion_vector == (0, 0, 0), "must settle before flipping direction"

        clock.tick(emergency_mod.SETTLE_S + 0.01)
        cmd = layer.evaluate(sensors)
        assert cmd.motion_vector[0] < 0, f"expected reverse: {cmd.message}"

        cmd = _advance_to_pivot(layer, sensors, clock)
        assert cmd.motion_vector[2] != 0, cmd.message
    print("PASS avoid backs off before pivoting when the rear is clear")


def test_avoid_skips_the_backoff_when_the_rear_is_tight():
    """No room behind -> pivot straight away rather than reverse into it."""
    with fake_emergency_clock() as clock:
        layer = EmergencyStopLayer()
        sensors = FakeDirectionalSensors(front=8, front_left=90, front_right=90,
                                          back=emergency_mod.BACKOFF_CLEARANCE_CM - 1)
        cmd = _advance_to_pivot(layer, sensors, clock)
        for _ in range(int(emergency_mod.BACKOFF_S / 0.1) + 2):
            assert cmd.motion_vector[0] >= 0, f"reversed into a tight rear: {cmd.message}"
            clock.tick(0.1)
            cmd = layer.evaluate(sensors)
    print("PASS avoid skips the backoff when the rear is tight")


def test_avoid_turns_toward_the_roomier_side():
    """Obstacle on the right must turn LEFT (+ve v_theta), and vice versa."""
    with fake_emergency_clock() as clock:
        layer = EmergencyStopLayer()
        cmd = _advance_to_pivot(
            layer, FakeDirectionalSensors(front=10, front_right=8, front_left=90), clock)
        assert cmd.motion_vector[2] > 0, cmd.message

    with fake_emergency_clock() as clock:
        layer = EmergencyStopLayer()
        cmd = _advance_to_pivot(
            layer, FakeDirectionalSensors(front=10, front_left=8, front_right=90), clock)
        assert cmd.motion_vector[2] < 0, cmd.message
    print("PASS avoid turns toward the roomier side")


def test_avoid_does_not_oscillate():
    """The reported bug: front_right trips -> turn left -> the obstacle slides
    into the front cone -> a fresh decision turned back right, forever."""
    with fake_emergency_clock() as clock:
        layer = EmergencyStopLayer()
        first = _advance_to_pivot(
            layer, FakeDirectionalSensors(front=20, front_right=8, front_left=90), clock)
        assert first.motion_vector[2] > 0, first.message

        # Same obstacle, now seen by the front sensor instead of the diagonal.
        for _ in range(5):
            clock.tick(0.1)
            cmd = layer.evaluate(FakeDirectionalSensors(front=8, front_right=20, front_left=90))
            assert cmd.motion_vector[2] > 0, f"reversed direction: {cmd.message}"
    print("PASS avoid holds its direction instead of oscillating")


def test_avoid_keeps_turning_until_clear_of_the_margin():
    """Barely clearing the trigger range must NOT end the turn -- that is what
    leaves the robot juddering on the threshold."""
    with fake_emergency_clock() as clock:
        layer = EmergencyStopLayer()
        _advance_to_pivot(
            layer, FakeDirectionalSensors(front=8, front_left=90, front_right=90), clock)
        clock.tick(emergency_mod.MIN_TURN_S + 0.1)

        just_outside = SensorHub.EMERGENCY_STOP_CM + 1
        cmd = layer.evaluate(FakeDirectionalSensors(front=just_outside,
                                                    front_left=90, front_right=90))
        assert cmd.active, "released the turn while still hugging the obstacle"

        clear = SensorHub.EMERGENCY_STOP_CM * emergency_mod.CLEAR_MARGIN + 1
        cmd = layer.evaluate(FakeDirectionalSensors(front=clear, front_left=90, front_right=90))
        assert not cmd.active, cmd.message
    print("PASS avoid keeps turning until clear of the margin")


def test_avoid_gives_up_when_wedged():
    with fake_emergency_clock() as clock:
        layer = EmergencyStopLayer()
        blocked = FakeDirectionalSensors(front=8, front_left=90, front_right=90)
        _advance_to_pivot(layer, blocked, clock)
        clock.tick(emergency_mod.MAX_TURN_S + 0.1)
        cmd = layer.evaluate(blocked)
        assert cmd.active and cmd.motion_vector == (0, 0, 0), cmd.message
    print("PASS avoid halts instead of spinning forever when wedged")


def test_wedged_stays_latched_instead_of_restarting_the_turn():
    """Live log: the turn timed out, then the very next tick started the same
    doomed turn again, forever. It must stay put until something changes."""
    with fake_emergency_clock() as clock:
        layer = EmergencyStopLayer()
        blocked = FakeDirectionalSensors(front=8, front_left=90, front_right=90)
        _advance_to_pivot(layer, blocked, clock)
        clock.tick(emergency_mod.MAX_TURN_S + 0.1)

        for _ in range(10):
            cmd = layer.evaluate(blocked)
            assert cmd.motion_vector == (0, 0, 0), f"restarted the turn: {cmd.message}"
            clock.tick(0.5)

        clear = SensorHub.EMERGENCY_STOP_CM * emergency_mod.CLEAR_MARGIN + 1
        cmd = layer.evaluate(FakeDirectionalSensors(front=clear, front_left=clear,
                                                     front_right=clear))
        assert not cmd.active, cmd.message
    print("PASS wedged stays latched until the readings genuinely change")


def test_scan_pivots_away_from_the_tighter_side():
    """Live log: the left side was hard up against a wall and the pattern
    pivoted left anyway -- the side came purely from the alternation flag."""
    class Sides:
        def __init__(self, left, right):
            self.left, self.right = left, right

        def get_obstacle_distance_front_left_cm(self):
            return self.left

        def get_obstacle_distance_front_right_cm(self):
            return self.right

    layer = ScanAroundLayer()
    layer._turn_left = True
    assert layer._pivot_side(Sides(left=16.0, right=None)) is False

    layer._turn_left = False
    assert layer._pivot_side(Sides(left=None, right=16.0)) is True

    # Alternation still owns the choice when neither side is the tighter one:
    # that alternation IS the zigzag, so a far wall must not cancel it.
    layer._turn_left = True
    assert layer._pivot_side(Sides(left=None, right=None)) is True
    assert layer._pivot_side(Sides(left=60.0, right=None)) is True

    # Both sides close: the wide old margin refused to flip here and the robot
    # pivoted into the nearer wall.
    layer._turn_left = True
    assert layer._pivot_side(Sides(left=12.0, right=18.0)) is False
    layer._turn_left = False
    assert layer._pivot_side(Sides(left=18.0, right=12.0)) is True
    print("PASS scan pivots away from the tighter side")


def test_scan_sees_the_diagonals():
    """A diagonal ends the lane only when it is genuinely close. At the front
    sensor's own threshold it must not: that is a wall being driven PAST, and
    treating it as a wall ahead trapped the robot spinning in a corner."""
    with fake_clock():
        layer = ScanAroundLayer()
        sensors = FakeDirectionalSensors(front=None, front_left=None,
                                          front_right=scan_mod.DIAGONAL_TURN_AT_CM - 1)
        sensors.litter = None
        layer.evaluate(sensors)                 # enter DRIVE
        cmd = layer.evaluate(sensors)
        assert "turn 1" in cmd.message, cmd.message

    with fake_clock():
        layer = ScanAroundLayer()
        sensors = FakeDirectionalSensors(front=None, front_left=None,
                                          front_right=TURN_AT_CM - 5)
        sensors.litter = None
        layer.evaluate(sensors)
        cmd = layer.evaluate(sensors)
        assert "lane" in cmd.message, f"a wall alongside ended the lane: {cmd.message}"
    print("PASS a diagonal ends the lane only when genuinely close")


def test_scan_drives_out_of_a_corner_instead_of_spinning():
    """Live log: front open, left open, right wall at 27cm -> the lane lasted
    exactly one tick and the pattern cycled turn/shift/turn forever."""
    with fake_clock() as clock:
        layer = ScanAroundLayer()
        sensors = FakeDirectionalSensors(front=None, front_left=None, front_right=27.0)
        sensors.litter = None
        layer.evaluate(sensors)
        for _ in range(20):
            cmd = layer.evaluate(sensors)
            assert "lane" in cmd.message, f"stopped driving: {cmd.message}"
            assert cmd.motion_vector[0] > 0 and cmd.motion_vector[2] == 0, cmd.message
            clock.tick(0.05)
    print("PASS scan drives out of a corner instead of spinning")


def test_tied_diagonals_alternate_instead_of_always_turning_right():
    """Live log: both diagonals read nothing (open space = no echo), which the
    old tie-break resolved as 'right' every single time."""
    dirs = []
    with fake_emergency_clock() as clock:
        layer = EmergencyStopLayer()
        open_ahead = FakeDirectionalSensors(front=8, front_left=None, front_right=None)
        for _ in range(4):
            cmd = _advance_to_pivot(layer, open_ahead, clock)
            dirs.append(cmd.motion_vector[2] > 0)
            clock.tick(emergency_mod.MAX_TURN_S + 0.1)   # time out
            layer.evaluate(open_ahead)                    # -> WEDGED
            layer._phase = None                           # simulate a fresh escape
    assert len(set(dirs)) == 2, f"never tried the other side: {dirs}"
    print("PASS tied diagonals alternate instead of always turning right")


def test_a_distant_wall_does_not_steer_the_escape():
    """Live log: the robot escaped right every time. front_left saw a harmless
    wall ~60cm off, front_right got no echo at all, and comparing raw ranges
    made 'no echo' win outright -- neither side is actually obstructed."""
    layer = EmergencyStopLayer()
    dirs = [layer._pick_direction(60.0, emergency_mod._FAR) for _ in range(4)]
    assert len(set(dirs)) == 2, f"a far wall still biased the escape: {dirs}"

    # A genuinely close obstacle must still pin the direction away from it.
    layer = EmergencyStopLayer()
    assert all(layer._pick_direction(10.0, emergency_mod._FAR) < 0 for _ in range(4))
    assert all(layer._pick_direction(emergency_mod._FAR, 10.0) > 0 for _ in range(4))
    print("PASS a distant wall does not steer the escape")


def test_boxed_in_halts():
    """Blocked ahead on both diagonals with no room behind -> nothing to do."""
    layer = EmergencyStopLayer()
    cmd = layer.evaluate(FakeDirectionalSensors(front=8, front_left=8, front_right=8, back=8))
    assert cmd.active and cmd.motion_vector == (0, 0, 0), cmd.message
    print("PASS boxed in halts")


def test_rear_obstacle_alone_is_ignored():
    """Halting for something behind while driving forward stranded the patrol.
    The rear only gates reversing."""
    layer = EmergencyStopLayer()
    cmd = layer.evaluate(FakeDirectionalSensors(back=8, front=90))
    assert not cmd.active, cmd.message
    print("PASS rear obstacle alone is ignored while driving forward")


def test_scan_backoff_aborts_on_a_close_rear():
    """Layer 1's backoff is the only phase that reverses, so it is the only
    one the rear sensor may cut short."""
    with fake_clock() as clock:
        layer = ScanAroundLayer()
        sensors = FakeDirectionalSensors(front=BACKOFF_AT_CM - 1,
                                         front_left=90, front_right=90, back=None)
        layer.evaluate(sensors)                        # DRIVE -> sees the wall
        cmd = layer.evaluate(sensors)                  # -> BACKOFF
        assert cmd.motion_vector[0] < 0, cmd.message

        sensors.back = scan_mod.BACKOFF_REAR_MIN_CM - 1
        clock.tick(0.05)
        cmd = layer.evaluate(sensors)
        assert cmd.motion_vector[0] >= 0, f"kept reversing into it: {cmd.message}"
    print("PASS scan backoff aborts on a close rear")


def test_scan_timers_pause_while_suppressed():
    """Layer 1's phases are timed open-loop, so a suppressed layer must not
    burn through them while a higher layer is driving the robot."""
    with fake_clock() as clock:
        layer = ScanAroundLayer()
        sensors = FakeSensors()
        sensors.dist = TURN_AT_CM - 1
        layer.evaluate(sensors)            # DRIVE
        layer.evaluate(sensors)            # -> TURN1
        layer.notify_arbitration(won=False)

        # Suppressed for longer than a full pivot: the phase must survive it.
        clock.tick(layer.turn_90_s * 3)
        cmd = layer.evaluate(sensors)
        assert "turn 1" in cmd.message, cmd.message

        layer.notify_arbitration(won=True)
        clock.tick(layer.turn_90_s + 0.01)
        cmd = layer.evaluate(sensors)
        assert "shifting" in cmd.message, cmd.message
    print("PASS scan timers pause while suppressed")


def test_median_filter_absorbs_a_dropped_ping():
    """Live log: a wall read 16cm, then 38cm, then 16cm again within a few
    ticks -- one dropped echo per sensor was swinging every threshold."""
    from collections import deque
    from src.hardware.sensors.ultrasonic_sensor import UltrasonicSensor, MEDIAN_WINDOW

    sensor = UltrasonicSensor.__new__(UltrasonicSensor)
    sensor._history = deque(maxlen=MEDIAN_WINDOW)

    for raw in (16.0, 16.0):
        sensor._record(raw)
    sensor._record(38.0)                    # one bad ping must not move it
    assert sensor.get_distance_cm() == 16.0, sensor.get_distance_cm()

    for raw in (90.0, 90.0, 90.0):          # a real move is still followed
        sensor._record(raw)
    assert sensor.get_distance_cm() == 90.0, sensor.get_distance_cm()

    for raw in (None, None, None):          # a genuinely empty view reads None
        sensor._record(raw)
    assert sensor.get_distance_cm() is None, sensor.get_distance_cm()
    print("PASS median filter absorbs a dropped ping")


# ── Plain runner (no pytest needed) ───────────────────────────────────────

ALL_TESTS = [
    test_full_zigzag_cycle,
    test_backoff_when_wall_seen_late,
    test_lane_timeout_without_wall,
    test_corner_wall_during_shift,
    test_yields_to_target_and_restarts,
    test_none_distance_is_not_an_obstacle,
    test_executor_mixing,
    test_executor_brakes_during_grab,
    test_executor_brake_falls_back_to_stop,
    test_arbitration_with_real_hub,
    test_avoid_backs_off_before_pivoting_when_the_rear_is_clear,
    test_avoid_skips_the_backoff_when_the_rear_is_tight,
    test_avoid_turns_toward_the_roomier_side,
    test_avoid_does_not_oscillate,
    test_avoid_keeps_turning_until_clear_of_the_margin,
    test_avoid_gives_up_when_wedged,
    test_rear_obstacle_alone_is_ignored,
    test_scan_backoff_aborts_on_a_close_rear,
    test_boxed_in_halts,
    test_a_distant_wall_does_not_steer_the_escape,
    test_tied_diagonals_alternate_instead_of_always_turning_right,
    test_wedged_stays_latched_instead_of_restarting_the_turn,
    test_scan_pivots_away_from_the_tighter_side,
    test_scan_sees_the_diagonals,
    test_scan_drives_out_of_a_corner_instead_of_spinning,
    test_median_filter_absorbs_a_dropped_ping,
    test_scan_timers_pause_while_suppressed,
]

if __name__ == "__main__":
    failed = []
    for fn in ALL_TESTS:
        try:
            fn()
        except AssertionError as exc:
            failed.append(fn.__name__)
            print(f"FAIL {fn.__name__}: {exc}")
    print()
    if failed:
        print(f"{len(failed)}/{len(ALL_TESTS)} FAILED: {', '.join(failed)}")
        sys.exit(1)
    print(f"all {len(ALL_TESTS)} tests passed")
