"""
tests/test_target_lock.py

Pure-logic tests for TargetLock (largest-bbox-first target priority).
NO hardware, no camera, no model — runs on Windows or the Pi:

    python tests/test_target_lock.py    # plain runner with PASS/FAIL summary
    pytest tests/test_target_lock.py    # also works

Covers:
    - picks the largest bbox area, not the highest confidence
    - keeps that tin while a bigger one appears (no re-aiming mid-approach)
    - tracks it across frames as the box grows/moves while driving in
    - holds through a short dropout (arm occludes camera during a grab)
    - releases after the grace window and re-picks the new largest
    - release() forces an immediate re-pick
    - a different tin nearby cannot steal the lock
    - CameraSensor's getters follow the lock and select once per new frame
"""
import importlib.util
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Loaded by path: importing src.perception pulls in the detector, which needs
# torch/ultralytics. TargetLock itself is pure math, so the lock tests below
# run on a plain Windows checkout; the CameraSensor tests skip without torch.
_spec = importlib.util.spec_from_file_location(
    "target_lock", project_root / "src" / "perception" / "target_lock.py")
_mod = importlib.util.module_from_spec(_spec) # type: ignore
_spec.loader.exec_module(_mod) # type: ignore
TargetLock, LOST_GRACE_S = _mod.TargetLock, _mod.LOST_GRACE_S


def _perception_available() -> bool:
    try:
        import src.perception.detector  # noqa: F401
        return True
    except Exception as exc:
        print(f"SKIP (no perception stack here: {exc})")
        return False




# ── Test doubles ──────────────────────────────────────────────────────────

class Box:
    """Minimal stand-in for perception.detector.BoundingBox."""

    def __init__(self, x1, y1, x2, y2, confidence=0.9, name=""):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.confidence = confidence
        self.name = name
        self.orientation = None

    @property
    def center_x(self):
        return (self.x1 + self.x2) // 2

    @property
    def center_y(self):
        return (self.y1 + self.y2) // 2

    @property
    def base_center(self):
        return (self.center_x, self.y2)


def square(cx, cy, size, **kw):
    h = size // 2
    return Box(cx - h, cy - h, cx + h, cy + h, **kw)


# ── Tests ─────────────────────────────────────────────────────────────────

def test_picks_largest_not_most_confident():
    big = square(300, 400, 200, confidence=0.80, name="big")
    small = square(900, 300, 60, confidence=0.99, name="small")
    lock = TargetLock()
    assert lock.select([small, big], now=0.0) is big, "largest bbox must win"


def test_ignores_bigger_newcomer_while_locked():
    tin = square(300, 400, 120, name="tin")
    lock = TargetLock()
    assert lock.select([tin], now=0.0) is tin
    huge = square(900, 500, 400, name="huge")
    picked = lock.select([tin, huge], now=0.2)
    assert picked is tin, "must stay on the locked tin, not jump to the bigger one"


def test_tracks_growing_box_while_driving_in():
    lock = TargetLock()
    tin = square(640, 300, 100)
    assert lock.select([tin], now=0.0) is tin
    t = 0.0
    for step in range(1, 6):
        t += 0.3
        tin = square(640 + step * 15, 300 + step * 25, 100 + step * 30)
        other = square(200, 200, 90)
        assert lock.select([other, tin], now=t) is tin, f"lost track at step {step}"


def test_holds_through_short_dropout():
    tin = square(500, 400, 150)
    lock = TargetLock()
    lock.select([tin], now=0.0)
    other = square(1000, 200, 300)
    # Missing but inside the grace window: no target, and NOT the other tin.
    assert lock.select([other], now=0.5) is None
    assert lock.locked, "lock must survive a short dropout"
    assert lock.select([tin, other], now=0.9) is tin, "must resume the same tin"


def test_releases_after_grace_and_repicks_largest():
    tin = square(500, 400, 150)
    lock = TargetLock()
    lock.select([tin], now=0.0)
    nxt = square(1000, 200, 90)
    far = square(100, 200, 70)
    t = LOST_GRACE_S + 0.1
    assert lock.select([nxt, far], now=t) is nxt, "after the grab, largest of the rest"


def test_release_forces_repick():
    tin = square(500, 400, 150)
    bigger = square(900, 400, 300)
    lock = TargetLock()
    lock.select([tin, bigger], now=0.0)
    lock.release()
    assert not lock.locked
    assert lock.select([tin, bigger], now=0.1) is bigger


def test_nearby_tin_cannot_steal_lock():
    tin = square(400, 400, 100)
    lock = TargetLock()
    lock.select([tin], now=0.0)
    # Same size, far to the side: outside the distance gate.
    intruder = square(900, 400, 100)
    assert lock.select([intruder], now=0.2) is None, "distance gate must reject it"


def test_camera_sensor_getters_follow_lock():
    """CameraSensor without a camera: inject results, check the getters."""
    if not _perception_available():
        return
    from src.hardware.sensors.camera_sensor import CameraSensor
    from src.perception.detector import DetectionResult
    import time

    sensor = CameraSensor.__new__(CameraSensor)   # no camera / model / thread
    import threading
    sensor._result_lock = threading.Lock()
    sensor._target_lock = TargetLock()
    sensor._selected_box = None
    sensor._selected_at = -1.0

    tin = square(400, 600, 200)
    huge = square(1000, 500, 400)
    result = DetectionResult(detections=[tin, huge], # type: ignore
                             frame_width=1280, frame_height=720)
    sensor._latest_result = result
    sensor._latest_at = time.monotonic()

    pos = sensor.get_litter_position()
    assert pos is not None
    assert abs(pos[0] - huge.center_x / 1280) < 1e-6, "largest tin first"
    ground = sensor.get_litter_ground_contact()
    assert abs(ground[1] - huge.y2 / 720) < 1e-6, "ground contact of the same tin" # type: ignore

    # Same frame queried three times -> one selection, no state churn.
    sensor.get_litter_position()
    assert sensor._selected_box is huge

    # New frame where the locked tin shrank and another one is now larger.
    huge2 = square(1010, 505, 380)
    other = square(300, 400, 500)
    sensor._latest_result = DetectionResult(detections=[other, huge2], # type: ignore
                                            frame_width=1280, frame_height=720)
    sensor._latest_at = time.monotonic()
    pos = sensor.get_litter_position()
    assert abs(pos[0] - huge2.center_x / 1280) < 1e-6, "must stay on the locked tin"

    sensor.release_target()
    sensor._latest_at = time.monotonic()
    pos = sensor.get_litter_position()
    assert abs(pos[0] - other.center_x / 1280) < 1e-6, "release -> re-pick largest"


def test_stale_result_reports_nothing():
    if not _perception_available():
        return
    from src.hardware.sensors.camera_sensor import CameraSensor, STALE_AFTER_S
    from src.perception.detector import DetectionResult
    import threading, time

    sensor = CameraSensor.__new__(CameraSensor)
    sensor._result_lock = threading.Lock()
    sensor._target_lock = TargetLock()
    sensor._selected_box = None
    sensor._selected_at = -1.0
    sensor._latest_result = DetectionResult(detections=[square(400, 600, 200)], # type: ignore
                                            frame_width=1280, frame_height=720)
    sensor._latest_at = time.monotonic() - (STALE_AFTER_S + 1.0)

    assert sensor.get_litter_position() is None
    assert sensor.get_litter_ground_contact() is None
    assert sensor.get_litter_pose() is None


# ── Plain runner (no pytest needed) ───────────────────────────────────────

ALL_TESTS = [
    test_picks_largest_not_most_confident,
    test_ignores_bigger_newcomer_while_locked,
    test_tracks_growing_box_while_driving_in,
    test_holds_through_short_dropout,
    test_releases_after_grace_and_repicks_largest,
    test_release_forces_repick,
    test_nearby_tin_cannot_steal_lock,
    test_camera_sensor_getters_follow_lock,
    test_stale_result_reports_nothing,
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