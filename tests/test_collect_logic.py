"""
tests/test_collect_logic.py

Pure-logic tests for Layer 3 (Collect Litter): arc-grasp solving, the grab
gates, and the ArmExecutor dispatch. NO hardware needed — runs on Windows
or the Pi:

    python tests/test_collect_logic.py   # plain runner with PASS/FAIL summary
    pytest tests/test_collect_logic.py   # also works

The arc solver runs against a REAL ArcGraspSolver loaded from a temp yaml
grid (so the interpolation path is exercised); the GraspPlanner is faked.
Covers:

    - tin inside the calibrated arc grid  -> 'grab_arc' + interpolated pose
    - tin outside the grid                -> inactive (Layer 2 keeps driving)
    - overshot past the nearest arc       -> active backoff, not a stand-down
    - the grab waits out GRAB_STABLE_S, and GRABBABLE_LATCH_S rides out a
      one-frame detection blink
    - the front ultrasonic vetoes a grab it reads as out of range
    - ground-contact point preferred over bbox center
    - Layer 2 steering signs and distance scaling
    - Layer 5 stands down for a tin the arm can actually reach
    - ArmExecutor: grab -> dump -> home, cooldown, stow idempotence
    - arbitration: collect(3) > approach(2) > scan(1), emergency(5) > all

For the on-robot version use tests/test_subsumption_live.py.
"""
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import src.subsumption.arm_executor as arm_exec_mod
from src.subsumption.arm_executor import ArmExecutor, GRAB_COOLDOWN_S
from src.subsumption.arbitrator import Arbitrator, ActionCommand
from src.subsumption.layers.layer0_idle import SystemIdleLayer
from src.subsumption.layers.layer1_scan import ScanAroundLayer
from src.subsumption.layers.layer2_approach import ApproachLitterLayer
import src.subsumption.layers.layer3_collect as collect_mod
from src.subsumption.layers.layer3_collect import (
    CollectLitterLayer, GRAB_STABLE_S, GRABBABLE_LATCH_S, GRAB_CONFIRM_CM,
    BACKOFF_SPEED,
)
import src.subsumption.layers.layer4_emergency as emergency_mod
from src.subsumption.layers.layer4_emergency import EmergencyStopLayer
from src.arm.arc_grasp import ArcGraspSolver, save_config, ch5_from_angle


# ── Test doubles ──────────────────────────────────────────────────────────

# Two-row grid, nx span 0.2..0.8 per row, ny band 0.4..0.8 (+/- tol 0.05).
TEST_ROWS = [
    {"ny": 0.4, "ny_tol": 0.05, "nx_tol": 0.05, "samples": [
        {"nx": 0.2, "arm": [60.0, 140.0, 70.0, 160.0, 90.0]},
        {"nx": 0.8, "arm": [140.0, 140.0, 70.0, 160.0, 90.0]},
    ]},
    {"ny": 0.8, "ny_tol": 0.05, "nx_tol": 0.05, "samples": [
        {"nx": 0.2, "arm": [60.0, 150.0, 80.0, 170.0, 90.0]},
        {"nx": 0.8, "arm": [140.0, 150.0, 80.0, 170.0, 90.0]},
    ]},
]

# One lying row over the same area; CH5 baseline 77 is deliberately NOT a
# real roll value, so tests can prove the angle formula overrode it.
LYING_ROWS = [
    {"ny": 0.6, "ny_tol": 0.1, "nx_tol": 0.05, "samples": [
        {"nx": 0.2, "arm": [60.0, 130.0, 60.0, 150.0, 77.0]},
        {"nx": 0.8, "arm": [140.0, 130.0, 60.0, 150.0, 77.0]},
    ]},
]


def make_solver(upright=True, lying=False) -> ArcGraspSolver:
    """Real solver, isolated temp calibration file (v3 layout)."""
    path = Path(tempfile.mkdtemp()) / "arc_grasp.yaml"
    save_config({
        "version": 3,
        "upright": {"rows": TEST_ROWS if upright else []},
        "lying": {"rows": LYING_ROWS if lying else []},
    }, path)
    return ArcGraspSolver(path)


class FakeSensors:
    def __init__(self):
        self.center = None       # get_litter_position()
        self.ground = None       # get_litter_ground_contact()
        self.pose = None         # get_litter_pose(): {'klass':..., 'angle':...}
        self.dist = None

    def get_litter_position(self):
        return self.center

    def get_aerial_trash_position(self):
        return None

    def get_litter_ground_contact(self):
        return self.ground

    def get_litter_pose(self):
        return self.pose

    def get_obstacle_distance_cm(self):
        return self.dist

    def has_obstacle(self):
        return self.dist is not None and self.dist < 10.0

    def get_litter_distance_cm(self):
        return None

    def get_litter_too_close(self):
        return False


class FakePlanner:
    """Records the GraspPlanner calls the ArmExecutor makes."""

    def __init__(self, arc_ok=True):
        self.arc_ok = arc_ok
        self.calls = []

    def goto(self, target, label=""):
        self.calls.append(("goto", target if isinstance(target, str) else "pose"))

    def collect(self, pose, tin_pose="upright", dump=True):
        self.calls.append(("arc", tuple(round(v, 1) for v in pose), tin_pose))
        return self.arc_ok

    def open_gripper(self):
        self.calls.append(("grip", "open"))

    def close_gripper(self):
        self.calls.append(("grip", "close"))

    def dump_to_bin(self):
        self.calls.append("dump")


class FakeClock:
    def __init__(self, start=1000.0):
        self.t = start

    def __call__(self):
        return self.t

    def tick(self, s):
        self.t += s


@contextmanager
def fake_exec_clock():
    clock = FakeClock()
    real = arm_exec_mod.time.monotonic
    arm_exec_mod.time.monotonic = clock
    try:
        yield clock
    finally:
        arm_exec_mod.time.monotonic = real


@contextmanager
def fake_collect_clock():
    clock = FakeClock()
    real = collect_mod.time.monotonic
    collect_mod.time.monotonic = clock
    try:
        yield clock
    finally:
        collect_mod.time.monotonic = real


def settle(layer, sensors, clock):
    """Hold the trigger past GRAB_STABLE_S, then return the real command.
    Every grab is gated on a continuously-stable reading, so a single
    evaluate() only ever yields the 'hold' that waits for one."""
    layer.evaluate(sensors)
    clock.tick(GRAB_STABLE_S + 0.01)
    return layer.evaluate(sensors)


@contextmanager
def fake_emergency_clock():
    clock = FakeClock()
    real = emergency_mod.time.monotonic
    emergency_mod.time.monotonic = clock
    try:
        yield clock
    finally:
        emergency_mod.time.monotonic = real


# ── Layer 3: arc grasp + the grab gates ───────────────────────────────────

def test_arc_grab_inside_grid():
    with fake_collect_clock() as clock:
        layer = CollectLitterLayer(arc_solver=make_solver())
        sensors = FakeSensors()
        sensors.ground = (0.5, 0.6)             # dead center of the test grid
        cmd = settle(layer, sensors, clock)
        assert cmd.active and cmd.arm_action == 'grab_arc', cmd.message
        assert cmd.motion_vector == (0, 0, 0)   # base halted for the grab
        pose = cmd.arm_params['pose']
        # Bilinear midpoint of the grid: CH1 = 100 (azimuth mid), CH2 = 145
        # (between the 140/150 rows) — proves real interpolation ran.
        assert abs(pose[0] - 100.0) < 0.2 and abs(pose[1] - 145.0) < 0.2, pose
    print("PASS arc grab inside grid (interpolated pose)")


def test_grab_waits_for_stability():
    """One good frame is not evidence: a blurred or half-occluded mask can
    land inside the band for an instant. The base is held (not handed back to
    Layer 2) while the reading settles, and only then does the arm fire."""
    with fake_collect_clock() as clock:
        layer = CollectLitterLayer(arc_solver=make_solver())
        sensors = FakeSensors()
        sensors.ground = (0.5, 0.6)

        cmd = layer.evaluate(sensors)
        assert cmd.active and cmd.arm_action == 'hold', cmd.message
        assert cmd.motion_vector == (0, 0, 0), cmd.message

        clock.tick(GRAB_STABLE_S - 0.05)        # still short of the window
        assert layer.evaluate(sensors).arm_action == 'hold'

        clock.tick(0.1)
        assert layer.evaluate(sensors).arm_action == 'grab_arc'

        # A tin that leaves and comes back starts the clock over.
        layer.reset()
        assert layer.evaluate(sensors).arm_action == 'hold'
    print("PASS grab waits out GRAB_STABLE_S")


def test_latch_rides_out_a_blink():
    """Live failure this prevents: one dropped detection handed the tin back
    to Layer 2, which drove forward again and pushed it out of the band."""
    with fake_collect_clock() as clock:
        layer = CollectLitterLayer(arc_solver=make_solver())
        sensors = FakeSensors()
        sensors.ground = (0.5, 0.6)
        settle(layer, sensors, clock)

        sensors.ground = None                   # one blank frame
        cmd = layer.evaluate(sensors)
        assert cmd.active and cmd.arm_action == 'hold', cmd.message

        clock.tick(GRABBABLE_LATCH_S + 0.1)     # genuinely gone now
        assert not layer.evaluate(sensors).active
    print("PASS latch rides out a one-frame blink")


def test_too_close_backs_off():
    """Overshooting the nearest calibrated arc used to return inactive, so
    Layer 2 kept closing in and made it worse until Layer 5 tripped."""
    with fake_collect_clock() as clock:
        layer = CollectLitterLayer(arc_solver=make_solver())
        sensors = FakeSensors()
        sensors.ground = (0.5, 0.95)            # below the grid's lowest arc
        cmd = layer.evaluate(sensors)
        assert cmd.active, cmd.message
        assert cmd.motion_vector == (BACKOFF_SPEED, 0, 0), cmd.motion_vector
        assert cmd.arm_action != 'grab_arc', cmd.arm_action

        # Backing off into the band grabs normally again.
        sensors.ground = (0.5, 0.6)
        assert settle(layer, sensors, clock).arm_action == 'grab_arc'
    print("PASS overshoot backs off instead of standing down")


def test_ultrasonic_vetoes_a_far_grab():
    """The front sensor is the independent check on the vision estimate. No
    reading is NOT a veto — an off-center tin sits outside its narrow beam."""
    with fake_collect_clock() as clock:
        layer = CollectLitterLayer(arc_solver=make_solver())
        sensors = FakeSensors()
        sensors.ground = (0.5, 0.6)

        sensors.dist = GRAB_CONFIRM_CM + 10.0
        layer.evaluate(sensors)
        clock.tick(GRAB_STABLE_S + 0.01)
        assert not layer.evaluate(sensors).active, "grabbed past the veto"

        sensors.dist = GRAB_CONFIRM_CM - 5.0
        assert settle(layer, sensors, clock).arm_action == 'grab_arc'

        sensors.dist = None                     # no echo -> vision decides
        assert settle(layer, sensors, clock).arm_action == 'grab_arc'
    print("PASS ultrasonic vetoes a far grab, missing echo does not")


def test_inactive_when_outside_the_grid():
    with fake_collect_clock():
        layer = CollectLitterLayer(arc_solver=make_solver())
        sensors = FakeSensors()
        sensors.ground = (0.5, 0.1)             # far above the grid's ny band
        assert not layer.evaluate(sensors).active   # Layer 2 keeps approaching
        sensors.ground = None
        sensors.center = None                   # no litter at all
        assert not layer.evaluate(sensors).active
    print("PASS inactive outside the calibrated grid")


def test_prefers_ground_contact_over_center():
    with fake_collect_clock() as clock:
        layer = CollectLitterLayer(arc_solver=make_solver())
        sensors = FakeSensors()
        sensors.center = (0.5, 0.1)             # bbox center: OUTSIDE the grid
        sensors.ground = (0.5, 0.6)             # ground contact: INSIDE
        cmd = settle(layer, sensors, clock)
        assert cmd.active and cmd.arm_action == 'grab_arc', cmd.message
    print("PASS ground-contact point preferred over bbox center")


def test_is_grabbable_does_not_disturb_the_clock():
    """Layer 5 and ArmExecutor both poll this every tick; if it advanced the
    stability clock it would either delay or short-circuit every grab."""
    with fake_collect_clock() as clock:
        layer = CollectLitterLayer(arc_solver=make_solver())
        sensors = FakeSensors()
        sensors.ground = (0.5, 0.6)
        for _ in range(5):
            assert layer.is_grabbable(sensors)
        clock.tick(GRAB_STABLE_S + 0.01)
        # Polling alone never started the clock, so this is still the first
        # stable frame, not a grab.
        assert layer.evaluate(sensors).arm_action == 'hold'
        assert not layer.is_grabbable(FakeSensors())
    print("PASS is_grabbable is pure")


# ── Lying tins: dual grids, CH5-from-angle, migration ─────────────────────

def test_ch5_anchor_interpolation():
    """User anchors (2026-07-08): straight (90deg) -> 90, sideways (0/180) -> 180."""
    for angle, expect in [(90, 90.0), (0, 180.0), (180, 180.0),
                          (45, 135.0), (135, 135.0), (190, 170.0)]:
        got = ch5_from_angle(angle)
        assert abs(got - expect) < 1e-6, f"angle {angle}: got {got}, want {expect}"
    print("PASS CH5 anchor interpolation (incl. mod-180 wrap)")


def test_v2_config_migrates_to_upright():
    """A pre-lying (v2, flat rows) file becomes the UPRIGHT grid, in place."""
    import yaml
    path = Path(tempfile.mkdtemp()) / "arc_grasp.yaml"
    save_config({"version": 2, "rows": TEST_ROWS}, path)
    solver = ArcGraspSolver(path)
    assert solver.ready and not solver.ready_for("lying")
    assert solver.solve(0.5, 0.6) is not None          # old grid still solves
    on_disk = yaml.safe_load(path.read_text())
    assert "upright" in on_disk and "rows" not in on_disk  # rewritten as v3
    print("PASS v2 config migrates to upright grid")


def test_lying_solve_overrides_ch5():
    solver = make_solver(upright=False, lying=True)
    for angle, expect in [(0, 180.0), (90, 90.0), (45, 135.0)]:
        pose = solver.solve(0.5, 0.6, pose="lying", angle_deg=angle)
        assert pose is not None and pose[4] == expect, (angle, pose)
        assert pose[0] == 100.0                         # CH1 still interpolated
    # axial (end-on, no measurable angle) -> straight-at-robot default
    pose = solver.solve(0.5, 0.6, pose="axial")
    assert pose[4] == 90.0, pose
    # the baseline CH5=77 from the calibration samples never leaks through
    print("PASS lying solve: CH5 from angle, baseline overridden")


def test_layer3_routes_lying_and_axial():
    with fake_collect_clock() as clock:
        layer = CollectLitterLayer(arc_solver=make_solver(upright=False, lying=True))
        sensors = FakeSensors()
        # Lying tins are referenced by the bbox CENTER (tracks the graspable
        # middle at any orientation) — the ground-contact point is deliberately
        # set OUTSIDE the lying grid to prove it is NOT what gets used.
        sensors.center = (0.5, 0.6)                         # inside lying grid
        sensors.ground = (0.5, 0.1)                         # outside — must be ignored

        sensors.pose = {"klass": "lying", "angle": 0.0}     # sideways
        cmd = settle(layer, sensors, clock)
        assert cmd.active and cmd.arm_action == 'grab_arc', cmd.message
        assert cmd.arm_params['pose'][4] == 180.0, cmd.arm_params
        assert cmd.arm_params['tin_pose'] == 'lying', cmd.arm_params
        assert "lying" in cmd.message, cmd.message

        sensors.pose = {"klass": "axial", "angle": 12.3}    # angle meaningless
        cmd = layer.evaluate(sensors)
        assert cmd.active and cmd.arm_params['pose'][4] == 90.0, cmd.arm_params
        assert cmd.arm_params['tin_pose'] == 'axial', cmd.arm_params

        # upright klass reads the GROUND point (0.5, 0.1): upright grid is
        # EMPTY anyway -> not grabbable. reset() clears the latch the lying
        # grabs left behind, which would otherwise hold the base.
        layer.reset()
        sensors.pose = {"klass": "upright", "angle": 90.0}
        assert not layer.evaluate(sensors).active
        # no pose info at all -> historical default = upright -> also inactive
        sensors.pose = None
        assert not layer.evaluate(sensors).active
    print("PASS layer3 routes lying/axial to the lying grid (by bbox center)")


def test_emergency_stands_down_for_a_grabbable_tin():
    """At grab range the front sensor is looking AT the tin. Without the
    exemption Layer 5 outvoted the grab and drove away from every can the
    robot got close enough to collect."""
    with fake_collect_clock(), fake_emergency_clock():
        collect = CollectLitterLayer(arc_solver=make_solver())
        emergency = EmergencyStopLayer(grab_zone_check=collect.is_grabbable)
        sensors = FakeSensors()
        sensors.dist = 8.0                      # well inside EMERGENCY_STOP_CM

        sensors.ground = None
        assert emergency.evaluate(sensors).active, "e-stop must fire for a wall"

        emergency.reset()
        sensors.ground = (0.5, 0.6)             # that obstacle is the tin
        assert not emergency.evaluate(sensors).active

        # A broken predicate must never disarm the e-stop.
        emergency.reset()
        emergency.grab_zone_check = lambda _s: (_ for _ in ()).throw(RuntimeError("boom"))
        assert emergency.evaluate(sensors).active
    print("PASS emergency stands down for a grabbable tin")


# ── Layer 2: steering signs ───────────────────────────────────────────────

def test_approach_steering():
    layer = ApproachLitterLayer()
    sensors = FakeSensors()

    sensors.center = (0.8, 0.5)             # tin right -> steer right (< 0)
    v = layer.evaluate(sensors).motion_vector
    assert v[2] < 0, v
    sensors.center = (0.2, 0.5)             # tin left -> steer left (> 0)
    v = layer.evaluate(sensors).motion_vector
    assert v[2] > 0, v

    sensors.center = (0.5, 0.2)             # far -> faster than close
    far_fwd = layer.evaluate(sensors).motion_vector[0]
    sensors.center = (0.5, 0.8)
    near_fwd = layer.evaluate(sensors).motion_vector[0]
    assert far_fwd > near_fwd > 0, (far_fwd, near_fwd)

    sensors.center = None
    assert not layer.evaluate(sensors).active
    print("PASS approach steering signs + distance scaling")


# ── ArmExecutor dispatch ──────────────────────────────────────────────────

def test_executor_grab_dump_home_and_cooldown():
    with fake_exec_clock() as clock:
        planner = FakePlanner()
        ex = ArmExecutor(planner=planner)
        # Boot must NOT move the arm — homing happens on the first command
        # after the operator starts the system (user requirement 2026-07-08).
        assert planner.calls == [], planner.calls

        grab = ActionCommand(3, True, (0, 0, 0), 'grab_arc', "",
                             {'pose': [100.0, 145.0, 75.0, 165.0, 90.0]})
        ex.execute(grab)
        # First command since boot: home first (unknown boot pose; goto()
        # asserts every channel), then collect (grab + dump, both inside
        # the fake's single "arc" call), then home.
        assert planner.calls == [("goto", "home"), ("grip", "open"),
                                 ("arc", (100.0, 145.0, 75.0, 165.0, 90.0), "upright"),
                                 ("goto", "home"), ("grip", "open")], planner.calls

        # Tin still visible next tick (mid-cooldown) -> must NOT re-grab.
        planner.calls.clear()
        ex.execute(grab)
        assert planner.calls == [], planner.calls

        # After the cooldown a retry is allowed.
        clock.tick(GRAB_COOLDOWN_S + 0.1)
        ex.execute(grab)
        assert planner.calls[0][0] == "arc", planner.calls
    print("PASS executor grab->dump->home + cooldown")


def test_executor_stow_idempotent_and_failed_grab():
    with fake_exec_clock() as clock:
        planner = FakePlanner(arc_ok=False)
        ex = ArmExecutor(planner=planner)
        assert planner.calls == []          # no movement at boot

        stow = ActionCommand(1, True, (0.5, 0, 0), 'stow', "")
        ex.execute(stow)                    # first command -> home once
        ex.execute(stow)                    # already home -> no extra moves
        assert planner.calls == [("goto", "home"), ("grip", "open")], planner.calls
        planner.calls.clear()

        # 'hold' is Layer 3 waiting for the reading to settle: same travel
        # pose as stow, and it must not start a grab.
        ex.execute(ActionCommand(3, True, (0, 0, 0), 'hold', ""))
        assert planner.calls == [], planner.calls

        # Refused grab (unreachable pose): no dump, arm still returns home.
        ex.execute(ActionCommand(3, True, (0, 0, 0), 'grab_arc', "",
                                 {'pose': [100.0, 145.0, 75.0, 165.0, 90.0]}))
        assert ("goto", "bin") not in planner.calls, planner.calls
        assert planner.calls[-2] == ("goto", "home"), planner.calls

        # After a grab the pose is dirty -> next stow really homes again,
        # but only ONCE.
        planner.calls.clear()
        clock.tick(GRAB_COOLDOWN_S + 0.1)
        ex.execute(stow)
        ex.execute(stow)
        assert planner.calls == [], planner.calls  # _grab already ended home
    print("PASS executor stow idempotence + refused grab does not dump")


def test_command_write_through():
    """User-reported bug (2026-07-08): typing 'c1 96.7' did NOTHING because
    the tracked pose already said 96.7 — but the physical arm had never
    moved. A commanded channel must ALWAYS be written to the servo, even
    when tracking claims it's already at the target."""
    from src.arm.grasp_planner import GraspPlanner
    p = GraspPlanner()                       # DummyServo kit on dev machines
    # Simulate tracking being wrong: physical servo somewhere else entirely.
    p.actuator.kit.servo[0].angle = 50.0
    assert p.arm[0] != 50.0                 # tracked pose disagrees
    p.jog_channel(0, 0.0)                    # command == tracked value (no-op diff)
    assert p.actuator.kit.servo[0].angle == p.arm[0], \
        "command equal to tracked pose must still be written through"

    # Full-pose moves assert every channel too (goto with tracking already
    # at home must still command the servos).
    p.actuator.kit.servo[2].angle = 10.0
    p.goto("home")
    assert p.actuator.kit.servo[2].angle == p.arm[2], \
        "full-pose move must write channels the tracker thinks are in place"

    # a second home must still assert the pose.
    p.actuator.kit.servo[3].angle = 5.0
    p.goto("home")
    assert p.actuator.kit.servo[3].angle == p.arm[3], \
        "a repeated home must still write through stale tracking"
    print("PASS commands write through stale tracking (jog + full pose + repeat home)")


def test_curved_arc_rows():
    """User-observed bug (2026-07-11): a constant-radius CH1 sweep projects
    as a CURVE (lower in the image at the edges), but rows were modelled as
    horizontal lines — 'grabbable' at the edges of the line was a lie, and
    the true reachable spot below the line was rejected. Rows now follow
    their samples' own ny."""
    import yaml as _yaml
    path = Path(tempfile.mkdtemp()) / "arc_grasp.yaml"
    save_config({
        "version": 3,
        "upright": {"rows": [{
            # center of the arc at ny=0.50; edges dip to ny=0.60
            "ny": 0.50, "ny_tol": 0.03, "nx_tol": 0.05, "samples": [
                {"nx": 0.2, "ny": 0.60, "arm": [60.0, 140.0, 70.0, 160.0, 90.0]},
                {"nx": 0.5, "ny": 0.50, "arm": [100.0, 140.0, 70.0, 160.0, 90.0]},
                {"nx": 0.8, "ny": 0.60, "arm": [140.0, 140.0, 70.0, 160.0, 90.0]},
            ]}]},
        "lying": {"rows": []},
    }, path)
    solver = ArcGraspSolver(path)

    assert solver.solve(0.5, 0.50) is not None      # center, on the curve
    assert solver.solve(0.8, 0.60) is not None      # edge, at its TRUE (lower) ny
    # Edge at the CENTER's height: the old flat line said grabbable here —
    # the user physically couldn't reach. The curve rejects it.
    assert solver.solve(0.8, 0.50) is None
    # Center at the EDGE's height: too close for the center of the arc.
    assert solver.solve(0.5, 0.60) is None
    # Halfway azimuth: curve interpolates (ny 0.55 there is ON the arc).
    assert solver.solve(0.65, 0.55) is not None

    # Legacy flat rows (samples without ny) still behave as before.
    assert make_solver().solve(0.5, 0.6) is not None
    print("PASS curved arc rows (edge dips honored, flat-line lies rejected)")


def test_row_attach_by_posture():
    """User-hit (2026-07-11): adding a LEFT-edge sample for the near arc got
    saved into the far row, because attachment used nearest-curve-ny and the
    edge dip isn't in the curve yet. One arc = one CH2-4 fold, so posture is
    the reliable key (user's own suggestion). Data below mirrors their yaml."""
    import test_arc_grasp as arc_tool          # sibling test-tool module

    far_row = {"ny": 0.8889, "samples": [
        {"nx": 0.5094, "ny": 0.8889, "arm": [95.0, 165.0, 20.0, 30.0, 90.0]},
        {"nx": 0.8992, "ny": 0.9486, "arm": [127.0, 165.0, 20.0, 30.0, 90.0]}]}
    near_row = {"ny": 0.8556, "samples": [
        {"nx": 0.5086, "ny": 0.8556, "arm": [95.0, 120.0, 40.0, 40.0, 90.0]},
        {"nx": 0.8727, "ny": 0.9000, "arm": [125.0, 130.0, 40.0, 40.0, 90.0]}]}
    rows = [far_row, near_row]

    # The exact mis-attach: left-edge click, arm folded like the NEAR family.
    row, d, note = arc_tool.choose_row_for_sample(
        rows, [65.0, 130.0, 40.0, 40.0, 90.0], 0.1078, 0.9264)
    assert row is near_row, "posture 130/40/40 belongs to the near arc"
    assert d <= 10.0 and not note, (d, note)

    # Identical postures on both rows -> ny decides (documented fallback).
    twin_a = {"ny": 0.5, "samples": [{"nx": 0.5, "ny": 0.5,
                                      "arm": [90.0, 150.0, 30.0, 30.0, 90.0]}]}
    twin_b = {"ny": 0.7, "samples": [{"nx": 0.5, "ny": 0.7,
                                      "arm": [90.0, 150.0, 30.0, 30.0, 90.0]}]}
    row, _d, note = arc_tool.choose_row_for_sample(
        [twin_a, twin_b], [90.0, 150.0, 30.0, 30.0, 90.0], 0.5, 0.72)
    assert row is twin_b and "ambiguous" in note, note
    print("PASS row attachment by CH2-4 posture (ny only breaks ties)")


def test_asymmetric_ny_band():
    """User-observed (2026-07-11): the valid spot is ON the arc or a little
    ABOVE it (farther); a tin BELOW the line (closer) gets overshot. The
    old symmetric +/-ny_tol band lied on the near side."""
    path = Path(tempfile.mkdtemp()) / "arc_grasp.yaml"
    save_config({
        "version": 3,
        "upright": {"rows": [{
            "ny": 0.50, "ny_tol": 0.05, "nx_tol": 0.05, "samples": [
                {"nx": 0.2, "ny": 0.50, "arm": [60.0, 140.0, 70.0, 160.0, 90.0]},
                {"nx": 0.8, "ny": 0.50, "arm": [140.0, 140.0, 70.0, 160.0, 90.0]},
            ]}]},
        "lying": {"rows": []},
    }, path)
    solver = ArcGraspSolver(path)

    assert solver.solve(0.5, 0.50) is not None      # on the line
    assert solver.solve(0.5, 0.46) is not None      # a little ABOVE: ok (far tol)
    assert solver.solve(0.5, 0.505) is not None     # hair below: detection jitter
    # Clearly BELOW the line (closer): the old +/-0.05 band accepted this,
    # the arm overshot the tin. Now rejected (near tol default 0.01).
    assert solver.solve(0.5, 0.53) is None
    print("PASS asymmetric band: above-line ok, below-line rejected")


def test_pose_persists_across_sessions():
    """User-reported (2026-07-11): 'h' in test_arc_live snapped to home at
    full speed. Root cause: every session started by ASSUMING home, so when
    the arm was really elsewhere the stepped ramp was a no-op and only the
    write-through fired (one full-speed jump). The tracked pose is now
    persisted after every move and reloaded on start, so ramps begin from
    the arm's true last-commanded position."""
    from src.arm.grasp_planner import GraspPlanner
    p = GraspPlanner()
    p.jog_channel(0, +7.0)                        # move + persist
    moved_to = p.arm[0]

    q = GraspPlanner()                            # "next session"
    assert q.arm[0] == moved_to, \
        "new session must resume the previous session's commanded pose"

    q.goto("home")                                # leave a clean state behind
    r = GraspPlanner()
    assert r.arm == q.arm and r.gripper == q.gripper
    print("PASS commanded pose persists across sessions (ramps start true)")


def test_grab_order_per_tin_pose():
    """LAST channel is the descent onto the tin — upright: CH2 shoulder;
    lying: CH3 elbow. Full sequence = open + pre-lift(1,2,3) + pose order
    + close + final shoulder lift(1); gripper steps through move_channel
    too, since an instant jaw move spikes current."""
    from src.arm.grasp_planner import GraspPlanner
    p = GraspPlanner()
    seq = []
    p.move_channel = lambda ch, v: seq.append(ch)     # spy on the channel order
    pose = [100.0, 140.0, 70.0, 160.0, 90.0]

    p.collect(pose, dump=False)                    # upright (default)
    assert seq == [5] + [1, 2, 3] + [0, 4, 2, 3, 1] + [5, 1], seq

    seq.clear()
    p.collect(pose, tin_pose="lying", dump=False)  # elbow (CH3=idx 2) LAST
    assert seq == [5] + [1, 2, 3] + [0, 4, 3, 1, 2] + [5, 1], seq

    seq.clear()
    p.collect(pose, tin_pose="axial", dump=False)  # axial grabs like lying
    assert seq == [5] + [1, 2, 3] + [0, 4, 3, 1, 2] + [5, 1], seq
    print("PASS grab order: upright shoulder-last, lying/axial elbow-last")


def test_gripper_angle_clamp():
    """User-hit (2026-07-11): the gripper linkage was built for a YF-6125MG;
    the MG996R swap has shorter travel, and commanding OPEN=0 deg parked it
    against its end-stop = silent stall that cooked two servos. Every write
    path must clamp CH6 through CHANNEL_ANGLE_LIMITS."""
    import src.hardware.actuators.pca9685_driver as drv
    act = drv.ArmActuator()
    saved = dict(drv.CHANNEL_ANGLE_LIMITS)
    try:
        drv.CHANNEL_ANGLE_LIMITS[5] = (20.0, 60.0)   # a calibrated safe window
        act.set_gripper_angle(0.0)                   # old OPEN command
        assert act.kit.servo[5].angle == 20.0        # clamped off the end-stop
        act.set_channel_angle(5, 180.0)
        assert act.kit.servo[5].angle == 60.0
        # stepped_move's writes clamp too (ramp targets included)
        drv.stepped_move(act, [90.0] * 6, [90.0] * 5 + [0.0],
                         step_deg=50.0, step_delay=0.01)
        assert act.kit.servo[5].angle == 20.0
        # arm channels (no limits configured) still span 0-180
        act.set_channel_angle(0, 0.0)
        assert act.kit.servo[0].angle == 0.0
    finally:
        drv.CHANNEL_ANGLE_LIMITS.clear()
        drv.CHANNEL_ANGLE_LIMITS.update(saved)
    print("PASS gripper clamp: no write path can command past the safe window")


def test_smooth_move_semantics():
    """stepped_move v2 (smooth streaming): same (step_deg, step_delay) pace
    as the old jump-and-sleep version — span/speed total duration — but
    executed as a fine-grained eased trajectory. Ends EXACTLY on target."""
    import time as _time
    from src.hardware.actuators.pca9685_driver import ArmActuator, stepped_move
    act = ArmActuator()                          # DummyServo kit on dev machines

    # 20 deg at (5 deg / 0.05 s) = 100 deg/s -> ~0.2 s total.
    t0 = _time.monotonic()
    stepped_move(act, [90.0] * 6, [110.0] + [90.0] * 5, step_deg=5.0, step_delay=0.05)
    dt = _time.monotonic() - t0
    assert 0.1 < dt < 0.6, f"expected ~0.2s (old pace preserved), got {dt:.3f}s"
    assert act.kit.servo[0].angle == 110.0       # lands exactly on target

    # instant channels bypass the ramp; no-op moves return immediately.
    stepped_move(act, [90.0] * 6, [90.0] * 5 + [40.0], instant=(5,))
    assert act.kit.servo[5].angle == 40.0
    t0 = _time.monotonic()
    stepped_move(act, [90.0] * 6, [90.0] * 6)
    assert _time.monotonic() - t0 < 0.05, "no-op move must not sleep"
    print("PASS smooth stepped_move keeps pace, lands exact, no-op is instant")


# ── Arbitration: the full ground-litter stack ─────────────────────────────

def test_arbitration_stack():
    with fake_emergency_clock() as clock, fake_collect_clock() as collect_clock:
        collect = CollectLitterLayer(arc_solver=make_solver())
        layers = [SystemIdleLayer(), ScanAroundLayer(), ApproachLitterLayer(),
                  collect, EmergencyStopLayer()]
        arb = Arbitrator()
        sensors = FakeSensors()

        def winner():
            for l in layers:
                arb.submit_command(l.evaluate(sensors))
            win = arb.get_winning_action()
            arb.clear()
            return win

        # Litter visible but NOT grabbable -> approach (2) beats scan (1).
        sensors.center = sensors.ground = (0.5, 0.1)
        assert winner().layer_id == 2

        # Tin inside the arc grid -> collect (3) halts the base and wins,
        # first holding it still until the reading has been stable long enough.
        sensors.center = sensors.ground = (0.5, 0.6)
        win = winner()
        assert win.layer_id == 3 and win.arm_action == 'hold', win.message
        collect_clock.tick(GRAB_STABLE_S + 0.01)
        win = winner()
        assert win.layer_id == 3 and win.arm_action == 'grab_arc', win.message

        # Obstacle inside emergency range -> emergency (5) beats even the grab.
        sensors.dist = 8.0
        assert winner().layer_id == 5

        # No litter, no obstacle -> scan patrols, but only once layer 5 has
        # finished its escape (settle -> backoff -> settle -> pivot, and the
        # pivot itself is held for MIN_TURN_S).
        sensors.center = sensors.ground = None
        sensors.dist = None
        for _ in range(40):
            if winner().layer_id == 1:
                break
            clock.tick(0.2)
            collect_clock.tick(0.2)     # lets Layer 3's blink latch expire
        else:
            raise AssertionError("layer 5 never handed back to scan")
    print("PASS arbitration: 5 > 3 > 2 > 1")


# ── Plain runner (no pytest needed) ───────────────────────────────────────

ALL_TESTS = [
    test_arc_grab_inside_grid,
    test_grab_waits_for_stability,
    test_latch_rides_out_a_blink,
    test_too_close_backs_off,
    test_ultrasonic_vetoes_a_far_grab,
    test_inactive_when_outside_the_grid,
    test_prefers_ground_contact_over_center,
    test_is_grabbable_does_not_disturb_the_clock,
    test_ch5_anchor_interpolation,
    test_v2_config_migrates_to_upright,
    test_lying_solve_overrides_ch5,
    test_layer3_routes_lying_and_axial,
    test_emergency_stands_down_for_a_grabbable_tin,
    test_approach_steering,
    test_executor_grab_dump_home_and_cooldown,
    test_executor_stow_idempotent_and_failed_grab,
    test_command_write_through,
    test_curved_arc_rows,
    test_row_attach_by_posture,
    test_asymmetric_ny_band,
    test_pose_persists_across_sessions,
    test_grab_order_per_tin_pose,
    test_gripper_angle_clamp,
    test_smooth_move_semantics,
    test_arbitration_stack,
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
