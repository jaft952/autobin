from __future__ import annotations
import enum
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

from src.perception.detector import DetectionResult

CENTERING_CONFIG_PATH = Path(__file__).parent / "config" / "centering_config.yaml"



class ChassisMove(enum.Enum):
    """Discrete movement suggestion for the (future) chassis layer."""
    HOLD = "hold"                    # tin is in the sweet spot — do not move
    FORWARD = "forward"              # tin too far ahead — drive forward
    BACKWARD = "backward"            # tin too close — back up
    TURN_LEFT = "turn_left"          # tin left of the sweet spot
    TURN_RIGHT = "turn_right"        # tin right of the sweet spot
    FORWARD_LEFT = "forward_left"
    FORWARD_RIGHT = "forward_right"
    BACKWARD_LEFT = "backward_left"
    BACKWARD_RIGHT = "backward_right"
    SEARCH = "search"                # no tin visible — chassis should scan/rotate


@dataclass
class CenteringStatus:
    """One tick of centering output."""
    aligned: bool                    # tin inside tolerance THIS frame
    stable: bool                     # aligned for >= stable_frames in a row -> grasp-ready
    move: ChassisMove                # discrete suggestion
    error_x: float                   # normalized error, + = tin is RIGHT of target
    error_y: float                   # normalized error, + = tin is BEHIND target (too close)
    motion_vector: Tuple[float, float, float]  # (forward, strafe=0, steer) suggestion
    target_px: Optional[Tuple[int, int]]       # smoothed tin center in pixels (None if lost)
    message: str = ""


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass
class CenteringConfig:

    target_x: float = 0.5
    target_y: float = 0.5

    tolerance: float = 0.01

    image_y_is_backward: int = 1     # +1: larger pixel y = closer to robot
    mirror_x: int = 1                # -1 if the image is mirrored left/right

    gain_forward: float = 1.0
    gain_steer: float = 1.0
    # Exponential smoothing of the detected center (0 = no smoothing, 0.6 = heavy).
    ema_alpha: float = 0.4
    # How many consecutive aligned frames before declaring "stable" (debounce).
    stable_frames: int = 5
    # How many consecutive missed frames before declaring the tin lost.
    lost_after: int = 10

    too_close_box_height: float = 0.95

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "CenteringConfig":
        cfg = cls()
        p = path or CENTERING_CONFIG_PATH
        if p.exists():
            import yaml
            data = yaml.safe_load(p.read_text()) or {}
            for key, value in data.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, value)
        return cfg


# ── Controller ───────────────────────────────────────────────────────────────

class IBVSCentering:


    def __init__(self, config: Optional[CenteringConfig] = None):

        self.config = config or CenteringConfig.load()
        self._smoothed: Optional[Tuple[float, float]] = None
        self._missed = 0
        self._aligned_streak = 0

    def reset(self):
        """Forget smoothing/debounce state (call when starting a new approach)."""
        self._smoothed = None
        self._missed = 0
        self._aligned_streak = 0

    # ── main tick ────────────────────────────────────────────────────────

    def update(self, detection: DetectionResult) -> CenteringStatus:
        cfg = self.config

        best = detection.best
        if best is None or detection.frame_width == 0:
            self._missed += 1
            if self._missed >= cfg.lost_after:
                self._smoothed = None
                self._aligned_streak = 0
            return CenteringStatus(
                aligned=False, stable=False, move=ChassisMove.SEARCH,
                error_x=0.0, error_y=0.0, motion_vector=(0.0, 0.0, 0.0),
                target_px=None,
                message="[IBVS] no tin detected — suggest chassis SEARCH (rotate/scan)",
            )
        self._missed = 0

        bx, by = best.base_center
       
        cx, cy = float(bx), float(by)
        if self._smoothed is None:
            self._smoothed = (cx, cy)
        else:
            a = cfg.ema_alpha
            self._smoothed = (a * self._smoothed[0] + (1 - a) * cx,
                              a * self._smoothed[1] + (1 - a) * cy)
        sx, sy = self._smoothed

        error_x = cfg.mirror_x * (sx / detection.frame_width - cfg.target_x)
        error_y = cfg.image_y_is_backward * (sy / detection.frame_height - cfg.target_y)

        
        box_h_frac = (best.height / detection.frame_height) if detection.frame_height else 0.0
        if box_h_frac >= cfg.too_close_box_height:
            error_y = max(error_y, cfg.tolerance + 0.2)

        aligned = math.hypot(error_x, error_y) <= cfg.tolerance
        self._aligned_streak = self._aligned_streak + 1 if aligned else 0
        stable = self._aligned_streak >= cfg.stable_frames

        move = self._classify(error_x, error_y, aligned)

  
        if aligned:
            motion = (0.0, 0.0, 0.0)
        else:
            forward = max(-1.0, min(1.0, -error_y * cfg.gain_forward))
            steer = max(-1.0, min(1.0, error_x * cfg.gain_steer))
            motion = (round(forward, 3), 0.0, round(steer, 3))

        if stable:
            msg = "[IBVS] tin CENTERED & stable — ready to hand over to arm servoing"
        elif aligned:
            msg = (f"[IBVS] tin centered ({self._aligned_streak}/{cfg.stable_frames}"
                   f" frames) — confirming...")
        else:
            msg = (f"[IBVS] off-center err=({error_x:+.2f},{error_y:+.2f}) — "
                   f"suggest chassis {move.value.upper()}")

        return CenteringStatus(
            aligned=aligned, stable=stable, move=move,
            error_x=round(error_x, 3), error_y=round(error_y, 3),
            motion_vector=motion, target_px=(int(sx), int(sy)), message=msg,
        )

    # ── helpers ──────────────────────────────────────────────────────────

    def _classify(self, error_x: float, error_y: float, aligned: bool) -> ChassisMove:

        if aligned:
            return ChassisMove.HOLD
        tol = self.config.tolerance
        lon = "forward" if error_y < -tol else "backward" if error_y > tol else ""
        lat = "left" if error_x < -tol else "right" if error_x > tol else ""
        if lon and lat:
            return ChassisMove(f"{lon}_{lat}")
        if lon:
            return ChassisMove(lon)
        if lat:
            return ChassisMove(f"turn_{lat}")
        # Inside per-axis tolerances but outside the radial one: nudge the larger axis.
        if abs(error_y) >= abs(error_x):
            return ChassisMove.FORWARD if error_y < 0 else ChassisMove.BACKWARD
        return ChassisMove.TURN_LEFT if error_x < 0 else ChassisMove.TURN_RIGHT
