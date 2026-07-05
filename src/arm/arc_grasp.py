"""
src/arm/arc_grasp.py

Empirical polar-coordinate grasping ("table IK") — the pragmatic replacement
for model-based IK while the kinematic model doesn't match the metal.

The camera is fixed to the arm base, so a tin's pixel position maps directly
to arm-base polar coordinates:

    pixel x (nx) -> azimuth -> CH1 only.  Base yaw is a pure rotation about the
                    vertical axis: the gripper tip sweeps an EXACT horizontal
                    arc, so this direction has no model error at all.
    pixel y (ny) -> radius  -> CH2..CH5 interpolated per-servo between
                    HAND-CALIBRATED grasp poses (near/mid/far).

Every calibrated pose was tuned on the real arm until the grab physically
worked, so sag, link lengths and zero offsets are all baked in. The solver
never touches the kinematic model.

Calibration lives in src/visual_servoing/config/centering_config.yaml under the
top-level `arc_grasp:` key (written by tests/test_arc_grasp.py; all other keys
in that yaml — target_x, lying_target_y, ... — are preserved on save):

    arc_grasp:
      azimuth:
      nx_center: 0.503      # tin pixel-x (normalized) at the arc's center
      ch1_center: 103.0     # CH1 command that grabs at that pixel
      ch1_per_nx: -35.0     # dCH1 per unit normalized-x (least-squares fit)
      ch1_min: 40.0         # safety clamps for CH1
      ch1_max: 165.0
      samples:              # raw (nx, ch1) calibration points, kept for re-fits
        - [0.503, 103.0]
    radii:                  # 1+ hand-tuned poses, sorted by ny at load time
      - ny: 0.62            # tin pixel-y (normalized) this pose grabs at
        ny_tol: 0.05        # how far from ny this row may serve on its own
        arm: [145.0, 75.0, 168.0, 90.0]    # CH2 CH3 CH4 CH5

With ONE radius row this degrades gracefully to your original idea: a single
arc, CH1-only (tin must sit within ny_tol of that row's ny). Add a second row
(near/far) and the strip between them becomes grabbable via interpolation.
"""

from __future__ import annotations
from copy import deepcopy
from pathlib import Path

# Calibration is stored INSIDE the existing visual-servoing config, under its
# own top-level `arc_grasp:` key. Saving is read-modify-write: every other key
# already in that yaml (target_x, lying_target_y, ...) is preserved untouched.
from src.visual_servoing.ibvs_centering import CENTERING_CONFIG_PATH

CONFIG_PATH = CENTERING_CONFIG_PATH
ARC_KEY = "arc_grasp"

# Seed config written on first run: the hand-tuned sweet point from
# tests/test_ibvs_centering.py (GRASP_ARM = [103, 145, 75, 168, 90]).
# ny / azimuth fit start unset -> solver reports not-ready until calibrated.
DEFAULT_CONFIG = {
    "azimuth": {
        "nx_center": None,
        "ch1_center": 103.0,
        "ch1_per_nx": None,
        "ch1_min": 40.0,
        "ch1_max": 165.0,
        "samples": [],
    },
    "radii": [
        {"ny": None, "ny_tol": 0.05, "arm": [145.0, 75.0, 168.0, 90.0]},
    ],
}


def _read_full_yaml(path: Path) -> dict:
    import yaml
    if Path(path).exists():
        return yaml.safe_load(Path(path).read_text()) or {}
    return {}


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Return the `arc_grasp:` section of the shared config (or a fresh default)."""
    arc = _read_full_yaml(path).get(ARC_KEY)
    return arc if arc else deepcopy(DEFAULT_CONFIG)


def save_config(cfg: dict, path: Path = CONFIG_PATH):
    """Write ONLY the `arc_grasp:` key; every other key in the yaml is preserved."""
    import yaml
    full = _read_full_yaml(path)
    full[ARC_KEY] = cfg
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(full, default_flow_style=None, sort_keys=False))


def fit_azimuth(samples) -> tuple | None:
    """Least-squares line ch1 = ch1_center + k * (nx - nx_center) through the
    (nx, ch1) samples. Returns (nx_center, ch1_center, ch1_per_nx) or None."""
    if len(samples) < 2:
        return None
    xs = [float(s[0]) for s in samples]
    ys = [float(s[1]) for s in samples]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    var = sum((x - mx) ** 2 for x in xs)
    if var < 1e-9:
        return None  # all samples at the same pixel — can't fit a slope
    k = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / var
    return (round(mx, 4), round(my, 2), round(k, 2))


class ArcGraspSolver:
    """solve(nx, ny) -> [CH1..CH5] servo commands, or None (not grabbable there)."""

    def __init__(self, path: Path = CONFIG_PATH):
        self.path = Path(path)
        self.reload()

    def reload(self):
        self.cfg = load_config(self.path)
        rows = [r for r in self.cfg.get("radii", []) if r.get("ny") is not None]
        self.radii = sorted(rows, key=lambda r: float(r["ny"]))

    # ── status ───────────────────────────────────────────────────────────
    @property
    def azimuth_ready(self) -> bool:
        az = self.cfg["azimuth"]
        return az.get("ch1_per_nx") is not None and az.get("nx_center") is not None

    @property
    def ready(self) -> bool:
        return self.azimuth_ready and len(self.radii) > 0

    def status(self) -> str:
        az = self.cfg["azimuth"]
        parts = [
            f"azimuth: {'OK' if self.azimuth_ready else 'NOT calibrated'}"
            f" ({len(az.get('samples') or [])} samples)",
            f"radii: {len(self.radii)} row(s)"
            + (f" ny={[round(float(r['ny']), 3) for r in self.radii]}" if self.radii else ""),
        ]
        return " | ".join(parts)

    # ── solving ──────────────────────────────────────────────────────────
    def solve(self, nx: float, ny: float):
        if not self.ready:
            return None
        az = self.cfg["azimuth"]
        ch1 = float(az["ch1_center"]) + float(az["ch1_per_nx"]) * (nx - float(az["nx_center"]))
        if not (float(az["ch1_min"]) <= ch1 <= float(az["ch1_max"])):
            return None
        arm = self._arm_for_ny(float(ny))
        if arm is None:
            return None
        return [round(ch1, 1)] + [round(float(a), 1) for a in arm]

    def _arm_for_ny(self, ny: float):
        rs = self.radii
        lo, hi = float(rs[0]["ny"]), float(rs[-1]["ny"])
        # outside the calibrated band (plus each end's own tolerance) -> not grabbable
        if ny < lo - float(rs[0].get("ny_tol", 0.05)):
            return None
        if ny > hi + float(rs[-1].get("ny_tol", 0.05)):
            return None
        ny = min(max(ny, lo), hi)          # clamp into the strip, then interpolate
        if len(rs) == 1:
            return rs[0]["arm"]
        for a, b in zip(rs, rs[1:]):       # find the bracketing pair
            if ny <= float(b["ny"]):
                t = (ny - float(a["ny"])) / (float(b["ny"]) - float(a["ny"]))
                return [(1 - t) * float(p) + t * float(q)
                        for p, q in zip(a["arm"], b["arm"])]
        return rs[-1]["arm"]
