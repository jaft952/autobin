"""
src/arm/arc_grasp.py

Empirical polar-coordinate grasping ("table IK") — the pragmatic alternative
to model-based IK: every calibrated pose was tuned on the real arm until the
grab physically worked, so sag, link lengths and zero offsets are baked in.
The solver never touches the kinematic model.

v2 DESIGN (2026-07-05) — a bilinear grid over the camera image:

    Each calibrated ARC (one distance/radius) is a ROW: its pixel-y (ny) plus
    3+ SAMPLES along the arc (left / middle / right), each storing the FULL
    hand-tuned [CH1..CH5] pose that grabs there.

    solve(nx, ny):
      1. find the two rows bracketing ny            (radius interpolation)
      2. inside each row, piecewise-linear interpolate ALL FIVE channels
         between the samples bracketing nx          (azimuth interpolation)
      3. blend the two rows' results by ny.

    Why per-row azimuth samples instead of the old single global CH1(nx)
    line: the camera is NOT mounted on the CH1 axis, so parallax makes the
    pixel-x <-> CH1 relation radius-dependent — near and far arcs need their
    own left/right calibration (also: each row's grabbable nx span is exactly
    the span you sampled, so per-radius camera-edge limits fall out for free).

Grabbable region = the strip between the calibrated rows (each row's own
ny_tol extends the band at the ends), horizontally within each row's sampled
nx span (+ nx_tol). Outside -> solve() returns None, honestly.

Calibration lives in ITS OWN file, src/arm/config/arc_grasp.yaml (one file
per calibration domain — IBVS keeps centering_config.yaml, pixel_to_arm has
pixel_to_arm.yaml). Written by tests/test_arc_grasp.py:

    version: 2
    rows:                       # one per calibrated arc, any order
      - ny: 0.62                # pixel-y (normalized) of this arc
        ny_tol: 0.05            # how far past this row the band extends
        nx_tol: 0.05            # how far past the sampled nx span to allow
        samples:                # 1+ per row; 3+ (left/mid/right) recommended
          - nx: 0.21
            arm: [64.0, 146.0, 75.0, 168.0, 90.0]    # CH1..CH5
          - nx: 0.50
            arm: [103.0, 145.0, 75.0, 168.0, 90.0]
          - nx: 0.79
            arm: [141.0, 146.0, 75.0, 168.0, 90.0]

Before 2026-07-05 the calibration lived inside centering_config.yaml under an
`arc_grasp:` key; load_config() migrates a v2 section from there automatically
(copy — the old key is left in place, delete it by hand when convenient).

With ONE row of ONE sample this degrades to the original single-spot grab.
"""

from __future__ import annotations
from copy import deepcopy
from pathlib import Path

# Own config file — no key-sharing with other subsystems anymore. The legacy
# location (inside centering_config.yaml under `arc_grasp:`) is only read once
# for migration.
from src.visual_servoing.ibvs_centering import CENTERING_CONFIG_PATH as LEGACY_CONFIG_PATH

CONFIG_PATH = Path(__file__).parent / "config" / "arc_grasp.yaml"
ARC_KEY = "arc_grasp"    # key inside the LEGACY shared yaml only

NY_TOL_DEFAULT = 0.05    # vertical band extension past the end rows
NX_TOL_DEFAULT = 0.05    # horizontal extension past a row's sampled span

DEFAULT_CONFIG = {
    "version": 2,
    "rows": [],              # empty -> solver reports not-ready until calibrated
}


def _read_full_yaml(path: Path) -> dict:
    import yaml
    if Path(path).exists():
        return yaml.safe_load(Path(path).read_text()) or {}
    return {}


def load_config(path: Path = CONFIG_PATH,
                legacy_path: Path = LEGACY_CONFIG_PATH) -> dict:
    """Return the v2 config from its own file (or a fresh default). If the
    own file doesn't exist yet but a v2 section is found at the legacy
    location (centering_config.yaml `arc_grasp:` key), it is copied over
    once. A pre-v2 layout (old `azimuth:`/`radii:`) is never migrated —
    recalibrate; it stays untouched in the legacy yaml as its own backup."""
    cfg = _read_full_yaml(path)
    if cfg and "rows" in cfg:
        return cfg
    legacy = _read_full_yaml(legacy_path).get(ARC_KEY)
    if legacy and "rows" in legacy:
        save_config(legacy, path)
        print(f"[arc_grasp] migrated calibration from {legacy_path} -> {path} "
              f"(old key left in place; delete it by hand when convenient)")
        return legacy
    return deepcopy(DEFAULT_CONFIG)


def save_config(cfg: dict, path: Path = CONFIG_PATH):
    """The file is wholly owned by arc_grasp now — plain overwrite."""
    import yaml
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(cfg, default_flow_style=None, sort_keys=False))


def _lerp_arm(a, b, t: float):
    return [(1.0 - t) * float(p) + t * float(q) for p, q in zip(a, b)]


class ArcGraspSolver:
    """solve(nx, ny) -> [CH1..CH5] servo commands, or None (not grabbable there)."""

    def __init__(self, path: Path = CONFIG_PATH):
        self.path = Path(path)
        self.reload()

    def reload(self):
        self.cfg = load_config(self.path)
        rows = []
        for r in self.cfg.get("rows", []):
            samples = sorted(
                (s for s in (r.get("samples") or []) if s.get("nx") is not None),
                key=lambda s: float(s["nx"]),
            )
            if r.get("ny") is not None and samples:
                rows.append({**r, "samples": samples})
        # sorted by ny: image top (far) first, image bottom (near) last
        self.rows = sorted(rows, key=lambda r: float(r["ny"]))

    # ── status ───────────────────────────────────────────────────────────
    @property
    def ready(self) -> bool:
        return len(self.rows) > 0

    def status(self) -> str:
        if not self.rows:
            return "arc_grasp v2: NOT calibrated (no rows — 'y' to start one)"
        parts = []
        for r in self.rows:
            ss = r["samples"]
            span = (f"nx {float(ss[0]['nx']):.2f}~{float(ss[-1]['nx']):.2f}"
                    if len(ss) > 1 else f"nx {float(ss[0]['nx']):.2f} only")
            parts.append(f"ny={float(r['ny']):.3f} ({len(ss)} sample(s), {span})")
        return f"arc_grasp v2: {len(self.rows)} row(s) | " + " | ".join(parts)

    # ── solving ──────────────────────────────────────────────────────────
    def solve(self, nx: float, ny: float):
        if not self.ready:
            return None
        arm = self._solve_grid(float(nx), float(ny))
        if arm is None:
            return None
        return [round(max(0.0, min(180.0, float(v))), 1) for v in arm]

    def _solve_grid(self, nx: float, ny: float):
        rs = self.rows
        lo, hi = float(rs[0]["ny"]), float(rs[-1]["ny"])
        if ny < lo - float(rs[0].get("ny_tol", NY_TOL_DEFAULT)):
            return None
        if ny > hi + float(rs[-1].get("ny_tol", NY_TOL_DEFAULT)):
            return None
        ny = min(max(ny, lo), hi)          # clamp into the strip
        if len(rs) == 1:
            return self._solve_row(rs[0], nx)
        for a, b in zip(rs, rs[1:]):       # find the bracketing row pair
            if ny <= float(b["ny"]):
                t = (ny - float(a["ny"])) / (float(b["ny"]) - float(a["ny"]))
                if t <= 1e-9:              # sitting ON row a: only a matters
                    return self._solve_row(a, nx)
                if t >= 1.0 - 1e-9:        # sitting ON row b: only b matters
                    return self._solve_row(b, nx)
                arm_a = self._solve_row(a, nx)
                arm_b = self._solve_row(b, nx)
                if arm_a is None or arm_b is None:
                    return None            # nx outside one row's sampled span
                return _lerp_arm(arm_a, arm_b, t)
        return self._solve_row(rs[-1], nx)

    @staticmethod
    def _solve_row(row, nx: float):
        """Piecewise-linear interpolation of ALL 5 channels along one arc."""
        ss = row["samples"]
        tol = float(row.get("nx_tol", NX_TOL_DEFAULT))
        lo, hi = float(ss[0]["nx"]), float(ss[-1]["nx"])
        if nx < lo - tol or nx > hi + tol:
            return None                    # outside this row's sampled span
        nx = min(max(nx, lo), hi)
        if len(ss) == 1:
            return [float(v) for v in ss[0]["arm"]]
        for a, b in zip(ss, ss[1:]):       # find the bracketing sample pair
            if nx <= float(b["nx"]):
                t = (nx - float(a["nx"])) / (float(b["nx"]) - float(a["nx"]))
                return _lerp_arm(a["arm"], b["arm"], t)
        return [float(v) for v in ss[-1]["arm"]]
