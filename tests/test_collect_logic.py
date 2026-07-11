"""
tests/test_collect_logic.py

Pure-logic tests for Layer 3 (Collect Litter): arc-grasp primary, model-IK
fallback, and the ArmExecutor dispatch. NO hardware needed — runs on Windows
or the Pi:

    python tests/test_collect_logic.py   # plain runner with PASS/FAIL summary
    pytest tests/test_collect_logic.py   # also works

The arc solver runs against a REAL ArcGraspSolver loaded from a temp yaml
grid (so the interpolation path is exercised), pixel_to_arm and the
GraspPlanner are faked. Covers:

    - tin inside the calibrated arc grid  -> 'grab_arc' + interpolated pose
    - tin outside the grid, p2a ready     -> 'grab_ik' + floor target
    - outside grid + p2a not calibrated   -> inactive (Layer 2 keeps driving)
    - IK radial gate: too far / too close / behind the arm -> inactive
    - ground-contact point preferred over bbox center
    - Layer 2 steering signs and distance scaling
    - ArmExecutor: grab -> dump -> home, cooldown, stow idempotence,
      failed IK grab does not dump
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
from src.subsumption.layers.layer3_collect import (
    CollectLitterLayer, IK_MIN_RADIUS_M, IK_MAX_RADIUS_M,
)
from src.subsumption.layers.layer5_emergency import EmergencyStopLayer
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


class FakeP2A:
    """pixel_to_arm stand-in: fixed resolution, scriptable transform."""

    def __init__(self, ready=True, point=None):
        self._ready = ready
        self.point = point                  # (x_m, y_m) returned by transform
        self.cfg = {"resolution": [1920, 1080]}
        self.last_uv = None

    @property
    def ready(self):
        return self._ready

    def transform(self, u, v):
        self.last_uv = (u, v)
        return self.point


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


class FakePlanner:
    """Records the GraspPlanner calls the ArmExecutor makes."""

    def __init__(self, arc_ok=True, ik_ok=True):
        self.arc_ok, self.ik_ok = arc_ok, ik_ok
        self.calls = []

    def home(self):
        self.calls.append("home")

    def grab_arc_pose(self, pose, tin_pose="upright"):
        self.calls.append(("arc", tuple(round(v, 1) for v in pose), tin_pose))
        return self.arc_ok

    def dump_to_bin(self):
        self.calls.append("dump")

    def move_to(self, xyz, **kw):
        self.calls.append(("move_to", tuple(round(v, 3) for v in xyz)))
        return self.ik_ok

    def control_gripper(self, action):
        self.calls.append(("grip", action))


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


# ── Layer 3: arc primary, IK fallback ─────────────────────────────────────

def test_arc_grab_inside_grid():
    layer = CollectLitterLayer(arc_solver=make_solver(), pixel_to_arm=FakeP2A(ready=False))
    sensors = FakeSensors()
    sensors.ground = (0.5, 0.6)             # dead center of the test grid
    cmd = layer.evaluate(sensors)
    assert cmd.active and cmd.arm_action == 'grab_arc', cmd.message
    assert cmd.motion_vector == (0, 0, 0)   # base halted for the grab
    pose = cmd.arm_params['pose']
    # Bilinear midpoint of the grid: CH1 = 100 (azimuth mid), CH2 = 145
    # (between the 140/150 rows) — proves real interpolation ran.
    assert abs(pose[0] - 100.0) < 0.2 and abs(pose[1] - 145.0) < 0.2, pose
    print("PASS arc grab inside grid (interpolated pose)")


def test_ik_fallback_outside_grid():
    p2a = FakeP2A(point=(0.05, 0.20))       # r ~ 0.206 m, inside the gate
    layer = CollectLitterLayer(arc_solver=make_solver(), pixel_to_arm=p2a)
    sensors = FakeSensors()
    sensors.ground = (0.5, 0.1)             # far above the grid's ny band
    cmd = layer.evaluate(sensors)
    assert cmd.active and cmd.arm_action == 'grab_ik', cmd.message
    assert cmd.arm_params['target_m'] == (0.05, 0.20)
    # Denormalization must use the CALIBRATION resolution.
    assert p2a.last_uv == (0.5 * 1920, 0.1 * 1080), p2a.last_uv
    print("PASS IK fallback outside grid")


def test_inactive_when_no_method_applies():
    layer = CollectLitterLayer(arc_solver=make_solver(), pixel_to_arm=FakeP2A(ready=False))
    sensors = FakeSensors()
    sensors.ground = (0.5, 0.1)             # outside grid, p2a uncalibrated
    cmd = layer.evaluate(sensors)
    assert not cmd.active                   # Layer 2 keeps approaching
    sensors.ground = None
    sensors.center = None                   # no litter at all
    assert not layer.evaluate(sensors).active
    print("PASS inactive when neither method applies")


def test_ik_radial_gate():
    layer = CollectLitterLayer(arc_solver=make_solver(),
                               pixel_to_arm=FakeP2A(point=None))
    sensors = FakeSensors()
    sensors.ground = (0.5, 0.1)             # always outside the arc grid here
    for point, why in [
        ((0.0, IK_MAX_RADIUS_M + 0.1), "too far"),
        ((0.0, IK_MIN_RADIUS_M - 0.02), "too close"),
        ((0.1, -0.2), "behind the arm"),
    ]:
        layer.p2a.point = point
        cmd = layer.evaluate(sensors)
        assert not cmd.active, f"should reject {why}: {cmd.message}"
    print("PASS IK radial gate (far/close/behind rejected)")


def test_prefers_ground_contact_over_center():
    layer = CollectLitterLayer(arc_solver=make_solver(), pixel_to_arm=FakeP2A(ready=False))
    sensors = FakeSensors()
    sensors.center = (0.5, 0.1)             # bbox center: OUTSIDE the grid
    sensors.ground = (0.5, 0.6)             # ground contact: INSIDE
    cmd = layer.evaluate(sensors)
    assert cmd.active and cmd.arm_action == 'grab_arc', cmd.message
    print("PASS ground-contact point preferred over bbox center")


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
    layer = CollectLitterLayer(arc_solver=make_solver(upright=False, lying=True),
                               pixel_to_arm=FakeP2A(ready=False))
    sensors = FakeSensors()
    # Lying tins are referenced by the bbox CENTER (tracks the graspable
    # middle at any orientation) — the ground-contact point is deliberately
    # set OUTSIDE the lying grid to prove it is NOT what gets used.
    sensors.center = (0.5, 0.6)                         # inside lying grid
    sensors.ground = (0.5, 0.1)                         # outside — must be ignored

    sensors.pose = {"klass": "lying", "angle": 0.0}     # sideways
    cmd = layer.evaluate(sensors)
    assert cmd.active and cmd.arm_action == 'grab_arc', cmd.message
    assert cmd.arm_params['pose'][4] == 180.0, cmd.arm_params
    assert cmd.arm_params['tin_pose'] == 'lying', cmd.arm_params
    assert "lying" in cmd.message, cmd.message

    sensors.pose = {"klass": "axial", "angle": 12.3}    # angle meaningless
    cmd = layer.evaluate(sensors)
    assert cmd.active and cmd.arm_params['pose'][4] == 90.0, cmd.arm_params
    assert cmd.arm_params['tin_pose'] == 'axial', cmd.arm_params

    # upright klass reads the GROUND point (0.5, 0.1): upright grid is
    # EMPTY anyway -> not grabbable
    sensors.pose = {"klass": "upright", "angle": 90.0}
    assert not layer.evaluate(sensors).active
    # no pose info at all -> historical default = upright -> also inactive
    sensors.pose = None
    assert not layer.evaluate(sensors).active
    print("PASS layer3 routes lying/axial to the lying grid (by bbox center)")


def test_layer3_lying_never_falls_back_to_ik():
    """Outside the lying grid, a lying tin must NOT trigger the IK grab
    (that path was only validated on standing tins) — upright still does."""
    p2a = FakeP2A(point=(0.05, 0.20))                   # would pass the gate
    layer = CollectLitterLayer(arc_solver=make_solver(upright=True, lying=True),
                               pixel_to_arm=p2a)
    sensors = FakeSensors()
    sensors.center = (0.5, 0.1)                         # outside BOTH grids
    sensors.ground = (0.5, 0.1)

    sensors.pose = {"klass": "lying", "angle": 45.0}
    assert not layer.evaluate(sensors).active           # no IK for lying

    sensors.pose = {"klass": "upright", "angle": 90.0}
    cmd = layer.evaluate(sensors)
    assert cmd.active and cmd.arm_action == 'grab_ik', cmd.message
    print("PASS lying never falls back to IK; upright still does")


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
        # First command since boot: force-home first (unknown boot pose;
        # FakePlanner has no force_home -> falls back to home()), then grab.
        assert planner.calls == ["home",
                                 ("arc", (100.0, 145.0, 75.0, 165.0, 90.0), "upright"),
                                 "dump", "home"], planner.calls

        # Tin still visible next tick (mid-cooldown) -> must NOT re-grab.
        planner.calls.clear()
        ex.execute(grab)
        assert planner.calls == [], planner.calls

        # After the cooldown a retry is allowed.
        clock.tick(GRAB_COOLDOWN_S + 0.1)
        ex.execute(grab)
        assert planner.calls[0][0] == "arc", planner.calls
    print("PASS executor grab->dump->home + cooldown")


def test_executor_stow_idempotent_and_failed_ik():
    with fake_exec_clock() as clock:
        planner = FakePlanner(ik_ok=False)
        ex = ArmExecutor(planner=planner)
        assert planner.calls == []          # no movement at boot

        stow = ActionCommand(1, True, (0.5, 0, 0), 'stow', "")
        ex.execute(stow)                    # first command -> force-home once
        ex.execute(stow)                    # already home -> no extra moves
        assert planner.calls == ["home"], planner.calls
        planner.calls.clear()

        # Unreachable IK grab: no dump, but the arm still returns home.
        ex.execute(ActionCommand(3, True, (0, 0, 0), 'grab_ik', "",
                                 {'target_m': (0.05, 0.28)}))
        assert "dump" not in planner.calls, planner.calls
        assert planner.calls[-1] == "home", planner.calls

        # After a grab the pose is dirty -> next stow really homes again,
        # but only ONCE.
        planner.calls.clear()
        clock.tick(GRAB_COOLDOWN_S + 0.1)
        ex.execute(stow)
        ex.execute(stow)
        assert planner.calls == [], planner.calls  # _grab already ended home
    print("PASS executor stow idempotence + failed IK grab does not dump")


def test_command_write_through():
    """User-reported bug (2026-07-08): typing 'c1 96.7' did NOTHING because
    the tracked pose already said 96.7 — but the physical arm had never
    moved. A commanded channel must ALWAYS be written to the servo, even
    when tracking claims it's already at the target."""
    from src.arm.grasp_planner import GraspPlanner
    p = GraspPlanner()                       # DummyServo kit on dev machines
    # Simulate tracking being wrong: physical servo somewhere else entirely.
    p.actuator.kit.servo[0].angle = 50.0
    assert p._arm[0] != 50.0                 # tracked pose disagrees
    p.jog_channel(0, 0.0)                    # command == tracked value (no-op diff)
    assert p.actuator.kit.servo[0].angle == p._arm[0], \
        "command equal to tracked pose must still be written through"

    # Full-pose moves assert every channel too (home() with tracking already
    # at home must still command the servos).
    p.actuator.kit.servo[2].angle = 10.0
    p.home()
    assert p.actuator.kit.servo[2].angle == p._arm[2], \
        "full-pose move must write channels the tracker thinks are in place"

    # force_home is now stepped (gentle) but must STILL assert the pose.
    p.actuator.kit.servo[3].angle = 5.0
    p.force_home()
    assert p.actuator.kit.servo[3].angle == p._arm[3], \
        "stepped force_home must still write through stale tracking"
    print("PASS commands write through stale tracking (jog + full pose + force_home)")


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
    moved_to = p._arm[0]

    q = GraspPlanner()                            # "next session"
    assert q._arm[0] == moved_to, \
        "new session must resume the previous session's commanded pose"

    q.force_home()                                # leave a clean state behind
    r = GraspPlanner()
    assert r._arm == q._arm and r._gripper == q._gripper
    print("PASS commanded pose persists across sessions (ramps start true)")


def test_grab_order_per_tin_pose():
    """User-specified approach orders (2026-07-11): the LAST channel is the
    descent onto the tin — upright: CH2 shoulder; lying: CH3 elbow.
    Sequence = pre-lift (1,2,3) + pose order + final shoulder lift (1)."""
    from src.arm.grasp_planner import GraspPlanner
    p = GraspPlanner()
    seq = []
    p._move_one = lambda ch, v: seq.append(ch)     # spy on the channel order
    p.control_gripper = lambda a: None
    pose = [100.0, 140.0, 70.0, 160.0, 90.0]

    p.grab_arc_pose(pose)                          # upright (default)
    assert seq == [1, 2, 3] + [0, 4, 2, 3, 1] + [1], seq

    seq.clear()
    p.grab_arc_pose(pose, tin_pose="lying")        # elbow (CH3=idx 2) LAST
    assert seq == [1, 2, 3] + [0, 4, 3, 1, 2] + [1], seq

    seq.clear()
    p.grab_arc_pose(pose, tin_pose="axial")        # axial grabs like lying
    assert seq == [1, 2, 3] + [0, 4, 3, 1, 2] + [1], seq
    print("PASS grab order: upright shoulder-last, lying/axial elbow-last")


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
    layers = [SystemIdleLayer(), ScanAroundLayer(), ApproachLitterLayer(),
              CollectLitterLayer(arc_solver=make_solver(),
                                 pixel_to_arm=FakeP2A(ready=False)),
              EmergencyStopLayer()]
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

    # Tin inside the arc grid -> collect (3) halts the base and wins.
    sensors.center = sensors.ground = (0.5, 0.6)
    win = winner()
    assert win.layer_id == 3 and win.arm_action == 'grab_arc', win.message

    # Obstacle inside emergency range -> emergency (5) beats even the grab.
    sensors.dist = 8.0
    assert winner().layer_id == 5

    # No litter, no obstacle -> scan patrols.
    sensors.center = sensors.ground = None
    sensors.dist = None
    assert winner().layer_id == 1
    print("PASS arbitration: 5 > 3 > 2 > 1")


# ── Plain runner (no pytest needed) ───────────────────────────────────────

ALL_TESTS = [
    test_arc_grab_inside_grid,
    test_ik_fallback_outside_grid,
    test_inactive_when_no_method_applies,
    test_ik_radial_gate,
    test_prefers_ground_contact_over_center,
    test_ch5_anchor_interpolation,
    test_v2_config_migrates_to_upright,
    test_lying_solve_overrides_ch5,
    test_layer3_routes_lying_and_axial,
    test_layer3_lying_never_falls_back_to_ik,
    test_approach_steering,
    test_executor_grab_dump_home_and_cooldown,
    test_executor_stow_idempotent_and_failed_ik,
    test_command_write_through,
    test_curved_arc_rows,
    test_asymmetric_ny_band,
    test_pose_persists_across_sessions,
    test_grab_order_per_tin_pose,
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
