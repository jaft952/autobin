from __future__ import annotations
from copy import deepcopy
from pathlib import Path

LEGACY_CONFIG_PATH = (Path(__file__).resolve().parents[1]
                      / "visual_servoing" / "config" / "centering_config.yaml")

CONFIG_PATH = Path(__file__).parent / "config" / "arc_grasp.yaml"
ARC_KEY = "arc_grasp"


NY_TOL_NEAR_DEFAULT = 0.01

NX_TOL_DEFAULT = 0.02
NY_TOL_DEFAULT = 0.02

POSES = ("upright", "lying")

BAND_IN = "in_band"
BAND_TOO_FAR = "too_far"
BAND_TOO_CLOSE = "too_close"
BAND_NX_OUTSIDE = "nx_outside"
BAND_NOT_CALIBRATED = "not_calibrated"

DEFAULT_CH5_ANCHORS = [[0.0, 180.0], [90.0, 90.0], [180.0, 180.0]]

DEFAULT_CONFIG = {
    "version": 3,
    "upright": {"rows": []},
    "lying": {"rows": [], "ch5_anchors": deepcopy(DEFAULT_CH5_ANCHORS)},
}


def _read_full_yaml(path: Path) -> dict:
    import yaml
    if Path(path).exists():
        return yaml.safe_load(Path(path).read_text()) or {}
    return {}


def _normalize_v3(cfg: dict) -> dict:
    """Make sure the config has both grids and the angle anchors."""
    cfg.setdefault("version", 3)
    cfg.setdefault("upright", {}).setdefault("rows", [])
    lying = cfg.setdefault("lying", {})
    lying.setdefault("rows", [])
    lying.setdefault("ch5_anchors", deepcopy(DEFAULT_CH5_ANCHORS))
    return cfg


def _wrap_v2(v2_cfg: dict) -> dict:
    """Convert an old v2 config into the v3 format."""
    return _normalize_v3({"version": 3, "upright": {"rows": v2_cfg.get("rows", [])}})


def load_config(path: Path = CONFIG_PATH,
                legacy_path: Path = LEGACY_CONFIG_PATH) -> dict:
    """Load the arc grasp config."""
    cfg = _read_full_yaml(path)
    if cfg and ("upright" in cfg or "lying" in cfg):
        return _normalize_v3(cfg)
    if cfg and "rows" in cfg:
        v3 = _wrap_v2(cfg)
        save_config(v3, path)
        print(f"[arc_grasp] migrated v2 -> v3 in {path} "
              f"(existing rows are now the UPRIGHT grid; lying grid empty)")
        return v3
    legacy = _read_full_yaml(legacy_path).get(ARC_KEY)
    if legacy and "rows" in legacy:
        v3 = _wrap_v2(legacy)
        save_config(v3, path)
        print(f"[arc_grasp] migrated calibration from {legacy_path} -> {path} "
              f"(old key left in place; delete it by hand when convenient)")
        return v3
    return deepcopy(DEFAULT_CONFIG)


def save_config(cfg: dict, path: Path = CONFIG_PATH):
    """Save the arc grasp config, overwriting the file."""
    import yaml
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(cfg, default_flow_style=None, sort_keys=False))


def _lerp_arm(a, b, t: float):
    return [(1.0 - t) * float(p) + t * float(q) for p, q in zip(a, b)]


def row_ny_at(row, nx: float) -> float:
    """Return the row's image height (ny) at position nx."""
    default = float(row["ny"])
    pts = sorted((float(s["nx"]), float(s.get("ny", default)))
                 for s in row["samples"])
    if nx <= pts[0][0]:
        return pts[0][1]
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        if nx <= x2:
            t = 0.0 if x2 == x1 else (nx - x1) / (x2 - x1)
            return y1 + (y2 - y1) * t
    return pts[-1][1]


def ch5_from_angle(angle_deg: float, anchors=None) -> float:
    """Wrist roll angle (CH5) for a lying can from its image angle."""
    pts = sorted((float(a), float(c)) for a, c in (anchors or DEFAULT_CH5_ANCHORS))
    a = float(angle_deg) % 180.0
    if a <= pts[0][0]:
        return max(0.0, min(180.0, pts[0][1]))
    for (a1, c1), (a2, c2) in zip(pts, pts[1:]):
        if a <= a2:
            t = 0.0 if a2 == a1 else (a - a1) / (a2 - a1)
            return max(0.0, min(180.0, c1 + (c2 - c1) * t))
    return max(0.0, min(180.0, pts[-1][1]))


def pose_and_angle(box):
    """Get the can pose and angle from a detection."""
    o = getattr(box, "orientation", None)
    if o is None or o.klass == "upright":
        return "upright", None
    if o.klass == "axial":
        return "lying", None
    return "lying", o.angle


class ArcGraspSolver:
    """Turns a can's image position into arm servo angles."""

    def __init__(self, path: Path = CONFIG_PATH):
        self.path = Path(path)
        self.reload()

    def reload(self):
        self.cfg = load_config(self.path)
        self._domains = {}
        for pose in POSES:
            rows = []
            for r in self.cfg.get(pose, {}).get("rows", []):
                samples = sorted(
                    (s for s in (r.get("samples") or []) if s.get("nx") is not None),
                    key=lambda s: float(s["nx"]),
                )
                if r.get("ny") is not None and samples:
                    rows.append({**r, "samples": samples})
            self._domains[pose] = sorted(rows, key=lambda r: float(r["ny"]))
        self._anchors = self.cfg.get("lying", {}).get("ch5_anchors") or DEFAULT_CH5_ANCHORS

    @property
    def rows(self):
        """Rows of the upright grid."""
        return self._domains["upright"]

    def rows_for(self, pose: str):
        return self._domains["lying" if pose in ("lying", "axial") else "upright"]

    @property
    def ready(self) -> bool:
        """True if the upright grid is calibrated."""
        return len(self._domains["upright"]) > 0

    def ready_for(self, pose: str) -> bool:
        return len(self.rows_for(pose)) > 0

    def status(self) -> str:
        parts = []
        for pose in POSES:
            rows = self._domains[pose]
            if not rows:
                parts.append(f"{pose}: NOT calibrated")
                continue
            spans = []
            for r in rows:
                ss = r["samples"]
                span = (f"nx {float(ss[0]['nx']):.2f}~{float(ss[-1]['nx']):.2f}"
                        if len(ss) > 1 else f"nx {float(ss[0]['nx']):.2f} only")
                spans.append(f"ny={float(r['ny']):.3f} ({len(ss)} sample(s), {span})")
            parts.append(f"{pose}: {len(rows)} row(s) | " + " | ".join(spans))
        return "arc_grasp v3 | " + "  ||  ".join(parts)

    def solve(self, nx: float, ny: float, pose: str = "upright",
              angle_deg: float | None = None):
        """Servo angles for a can at (nx, ny), or None if out of reach."""
        return self.solve_with_band(nx, ny, pose, angle_deg)[0]

    def solve_with_band(self, nx: float, ny: float, pose: str = "upright",
                        angle_deg: float | None = None):
        """Like solve(), but also returns why the can is out of reach."""
        lying = pose in ("lying", "axial")
        rows = self.rows_for(pose)
        if not rows:
            return None, BAND_NOT_CALIBRATED
        arm, band = self._solve_grid(rows, float(nx), float(ny))
        if arm is None:
            return None, band
        if lying:
            arm[4] = ch5_from_angle(90.0 if angle_deg is None else angle_deg,
                                    self._anchors)
        return [round(max(0.0, min(180.0, float(v))), 1) for v in arm], band

    def _solve_grid(self, rs, nx: float, ny: float):
        """Solve against one grid and return (angles, band)."""
        curves = sorted(((row_ny_at(r, nx), r) for r in rs), key=lambda p: p[0])
        lo_ny, lo_row = curves[0]
        hi_ny, hi_row = curves[-1]
        if ny < lo_ny - float(lo_row.get("ny_tol", NY_TOL_DEFAULT)):
            return None, BAND_TOO_FAR
        if ny > hi_ny + float(hi_row.get("ny_tol_near", NY_TOL_NEAR_DEFAULT)):
            return None, BAND_TOO_CLOSE
        ny = min(max(ny, lo_ny), hi_ny)
        if len(curves) == 1:
            return self._banded(self._solve_row(lo_row, nx))
        for (a_ny, a_row), (b_ny, b_row) in zip(curves, curves[1:]):
            if ny <= b_ny:
                if b_ny - a_ny <= 1e-9:
                    return self._banded(self._solve_row(a_row, nx))
                t = (ny - a_ny) / (b_ny - a_ny)
                if t <= 1e-9:
                    return self._banded(self._solve_row(a_row, nx))
                if t >= 1.0 - 1e-9:
                    return self._banded(self._solve_row(b_row, nx))
                arm_a = self._solve_row(a_row, nx)
                arm_b = self._solve_row(b_row, nx)
                if arm_a is None or arm_b is None:
                    return None, BAND_NX_OUTSIDE
                return _lerp_arm(arm_a, arm_b, t), BAND_IN
        return self._banded(self._solve_row(curves[-1][1], nx))

    @staticmethod
    def _banded(arm):
        """Turn one row's answer into (angles, band)."""
        return (arm, BAND_IN) if arm is not None else (None, BAND_NX_OUTSIDE)

    @staticmethod
    def _solve_row(row, nx: float):
        """Interpolate all five channels along one arc."""
        ss = row["samples"]
        tol = float(row.get("nx_tol", NX_TOL_DEFAULT))
        lo, hi = float(ss[0]["nx"]), float(ss[-1]["nx"])
        if nx < lo - tol or nx > hi + tol:
            return None
        nx = min(max(nx, lo), hi)
        if len(ss) == 1:
            return [float(v) for v in ss[0]["arm"]]
        for a, b in zip(ss, ss[1:]):
            if nx <= float(b["nx"]):
                t = (nx - float(a["nx"])) / (float(b["nx"]) - float(a["nx"]))
                return _lerp_arm(a["arm"], b["arm"], t)
        return [float(v) for v in ss[-1]["arm"]]
