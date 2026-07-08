"""
src/arm/pixel_to_arm.py

Pixel -> arm-frame ground transform: "the camera sees the tin at pixel (u, v)
— where is it in metres, in the arm's coordinate frame?"

The tin always stands on the FLOOR, so this is a plane-to-plane mapping: the
detection's ground-contact pixel (BoundingBox.base_center) relates to the
floor point (x, y) by a single 3x3 homography. No camera intrinsics, no full
hand-eye calibration — 4+ ruler-measured correspondences pin it down.

Frames / conventions:
  * pixel (u, v): raw pixels in the SAME capture resolution used at runtime
    (the C270 crops differently per resolution, so calibration is only valid
    at the resolution it was captured at — stored and checked).
  * arm (x, y): metres on the floor. Same frame the IK uses: x=right,
    y=forward, origin = the point on the FLOOR directly under the CH1 yaw
    axis. The floor plane itself is z = -DECK_ABOVE_FLOOR_M in IK coords.
  * The camera is bolted to the chassis, the arm base too, so the mapping is
    constant — until the camera is physically moved. Then RECALIBRATE.

Calibrate with tests/test_pixel_grasp.py (place tin at measured spots, press
'c', then 'f' to fit+save). Calibration lives in ITS OWN file,
src/arm/config/pixel_to_arm.yaml (one file per calibration domain). A
calibration saved at the pre-2026-07-05 location (centering_config.yaml,
`pixel_to_arm:` key) is migrated automatically on first load (copied; the
old key is left in place, delete it by hand when convenient).

The fit is a normalised-DLT least-squares homography in pure numpy (no cv2),
so this module imports and unit-tests anywhere.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date
from pathlib import Path

# Literal legacy path instead of importing it from ibvs_centering: that
# import drags in the whole perception chain (YOLO/torch) just for a frozen
# filename, and this module is documented to import anywhere.
LEGACY_CONFIG_PATH = (Path(__file__).resolve().parents[1]
                      / "visual_servoing" / "config" / "centering_config.yaml")

CONFIG_PATH = Path(__file__).parent / "config" / "pixel_to_arm.yaml"
P2A_KEY = "pixel_to_arm"    # key inside the LEGACY shared yaml only

MIN_SAMPLES = 4          # a homography has 8 DoF -> 4 point pairs minimum

DEFAULT_CONFIG = {
    "homography": None,      # 3x3 nested list, pixel -> metres on the floor
    "resolution": None,      # [width, height] the samples were captured at
    "samples": [],           # [{u, v, x, y}] pixels + measured metres
    "fitted": None,          # ISO date of the last successful fit
}


def _read_full_yaml(path: Path) -> dict:
    import yaml
    if Path(path).exists():
        return yaml.safe_load(Path(path).read_text()) or {}
    return {}


def load_config(path: Path = CONFIG_PATH,
                legacy_path: Path = LEGACY_CONFIG_PATH) -> dict:
    """Return the calibration from its own file (or a default). If the own
    file doesn't exist yet but a calibration is found at the legacy location
    (centering_config.yaml `pixel_to_arm:` key), it is copied over once."""
    cfg = _read_full_yaml(path)
    if cfg and ("homography" in cfg or cfg.get("samples")):
        return cfg
    legacy = _read_full_yaml(legacy_path).get(P2A_KEY)
    if legacy:
        save_config(legacy, path)
        print(f"[pixel_to_arm] migrated calibration from {legacy_path} -> {path} "
              f"(old key left in place; delete it by hand when convenient)")
        return legacy
    return deepcopy(DEFAULT_CONFIG)


def save_config(cfg: dict, path: Path = CONFIG_PATH):
    """The file is wholly owned by pixel_to_arm now — plain overwrite."""
    import yaml
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(cfg, default_flow_style=None, sort_keys=False))


def _fit_homography(pixel_pts, arm_pts):
    """Normalised-DLT least-squares homography (pixels -> metres). Pure numpy.
    Returns a 3x3 numpy array with H[2,2] == 1."""
    import numpy as np

    px = np.asarray(pixel_pts, dtype=float)
    mt = np.asarray(arm_pts, dtype=float)

    def norm_transform(pts):
        # Hartley normalisation: centroid to origin, mean distance sqrt(2).
        c = pts.mean(axis=0)
        d = np.sqrt(((pts - c) ** 2).sum(axis=1)).mean()
        s = np.sqrt(2.0) / max(d, 1e-12)
        return np.array([[s, 0.0, -s * c[0]],
                         [0.0, s, -s * c[1]],
                         [0.0, 0.0, 1.0]])

    Tp, Tm = norm_transform(px), norm_transform(mt)
    ph = np.c_[px, np.ones(len(px))] @ Tp.T
    mh = np.c_[mt, np.ones(len(mt))] @ Tm.T

    rows = []
    for (u, v, w), (x, y, _) in zip(ph, mh):
        rows.append([u, v, w, 0, 0, 0, -x * u, -x * v, -x * w])
        rows.append([0, 0, 0, u, v, w, -y * u, -y * v, -y * w])
    _, _, vt = np.linalg.svd(np.asarray(rows))
    Hn = vt[-1].reshape(3, 3)

    H = np.linalg.inv(Tm) @ Hn @ Tp
    return H / H[2, 2]


class PixelToArm:
    """Holds the calibration; add_sample/fit/save during calibration,
    transform() at runtime."""

    def __init__(self, path: Path = CONFIG_PATH):
        self.path = Path(path)
        self.cfg = load_config(self.path)

    # ── status ───────────────────────────────────────────────────────────
    @property
    def ready(self) -> bool:
        return self.cfg.get("homography") is not None

    @property
    def samples(self) -> list:
        return self.cfg.setdefault("samples", [])

    def status(self) -> str:
        n = len(self.samples)
        if self.ready:
            return (f"pixel_to_arm: CALIBRATED ({n} samples, fitted {self.cfg.get('fitted')}, "
                    f"resolution {self.cfg.get('resolution')})")
        return f"pixel_to_arm: NOT calibrated ({n}/{MIN_SAMPLES} samples so far)"

    def check_resolution(self, width: int, height: int) -> bool:
        """True if the stored calibration matches this capture resolution."""
        res = self.cfg.get("resolution")
        return res is None or (int(res[0]) == int(width) and int(res[1]) == int(height))

    # ── calibration ──────────────────────────────────────────────────────
    def add_sample(self, u: float, v: float, x_m: float, y_m: float):
        self.samples.append({"u": float(u), "v": float(v),
                             "x": float(x_m), "y": float(y_m)})

    def undo_sample(self):
        if self.samples:
            return self.samples.pop()
        return None

    def fit(self, width: int = None, height: int = None):
        """Fit the homography from the stored samples. Returns per-sample
        residuals in centimetres, or None if not enough samples."""
        if len(self.samples) < MIN_SAMPLES:
            return None
        px = [(s["u"], s["v"]) for s in self.samples]
        mt = [(s["x"], s["y"]) for s in self.samples]
        H = _fit_homography(px, mt)
        self.cfg["homography"] = [[float(v) for v in row] for row in H]
        if width is not None and height is not None:
            self.cfg["resolution"] = [int(width), int(height)]
        self.cfg["fitted"] = date.today().isoformat()
        return self.residuals_cm()

    def residuals_cm(self) -> list:
        """|predicted - measured| per sample, in cm. Empty if not fitted."""
        if not self.ready:
            return []
        import numpy as np
        out = []
        for s in self.samples:
            p = self.transform(s["u"], s["v"])
            if p is None:
                out.append(float("nan"))
                continue
            out.append(float(np.hypot(p[0] - s["x"], p[1] - s["y"])) * 100.0)
        return out

    def save(self):
        save_config(self.cfg, self.path)

    # ── runtime ──────────────────────────────────────────────────────────
    def transform(self, u: float, v: float):
        """Ground-contact pixel -> (x, y) metres on the floor in the arm
        frame. None if not calibrated (or degenerate)."""
        if not self.ready:
            return None
        import numpy as np
        H = np.asarray(self.cfg["homography"], dtype=float)
        p = H @ np.array([float(u), float(v), 1.0])
        if abs(p[2]) < 1e-9:
            return None
        return (float(p[0] / p[2]), float(p[1] / p[2]))
