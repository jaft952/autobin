"""
src/visual_servoing/ibvs_centering.py

IBVS Stage 1 — Chassis Centering (placeholder until chassis control is merged).

Pipeline position:
    [THIS] IBVS centering: drive the CHASSIS until the tin sits in the camera's
           "grasp sweet spot" (the image region where the arm grasps comfortably).
    [NEXT] closed_loop_servo.py: fine ARM servoing (IK + pixel feedback) + grasp.

Camera geometry assumption: the camera is mounted high (tall wheels) looking
STRAIGHT DOWN, so image coordinates map approximately linearly to ground
positions around the robot. By default the TOP of the image is the robot's
FORWARD direction — if your camera is mounted rotated, set the flip flags in
CenteringConfig instead of editing the logic.

Chassis movement belongs to a teammate (not merged yet), so this module does
NOT drive any motors. It only classifies "is the tin in the sweet spot?" and
emits a movement SUGGESTION (discrete ChassisMove + a continuous
(forward, steer) vector in the same convention as subsumption ActionCommand
motion vectors) for the chassis layer to consume later.

Usage:
    from src.perception.detector import AluminiumCanDetector
    from src.visual_servoing.ibvs_centering import IBVSCentering

    detector = AluminiumCanDetector(...)
    centering = IBVSCentering()
    while True:
        status = centering.update(detector.detect())
        print(status.message)
        if status.stable:
            break   # hand over to ClosedLoopServo for the arm approach
"""

from __future__ import annotations
import enum
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

from src.perception.detector import DetectionResult

# Sweet-spot calibration is saved here by calibrate_sweet_spot.py and loaded by
# CenteringConfig.load(). See src/visual_servoing/config/centering_config.yaml.
CENTERING_CONFIG_PATH = Path(__file__).parent / "config" / "centering_config.yaml"


# ── Output types ─────────────────────────────────────────────────────────────

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
    # Sweet spot center, normalized image coords (0..1). Calibrate by placing a
    # tin at the arm's comfortable grasp point and reading its pixel center with
    # tests/test_ibvs_centering.py --live, then divide by frame size.
    target_x: float = 0.5
    target_y: float = 0.5
    # Normalized radius around the sweet spot that still counts as "centered".
    tolerance: float = 0.08
    # Camera mounting: top of image = robot forward (+1). Use -1 if mounted
    # rotated 180°; swap left/right with mirror_x = -1.
    image_y_is_backward: int = 1     # +1: larger pixel y = closer to robot
    mirror_x: int = 1                # -1 if the image is mirrored left/right
    # Suggestion gains: normalized error -> (forward, steer) magnitudes (unitless,
    # 0..~1, the chassis layer decides what they mean physically).
    gain_forward: float = 1.0
    gain_steer: float = 1.0
    # Exponential smoothing of the detected center (0 = no smoothing, 0.6 = heavy).
    ema_alpha: float = 0.4
    # How many consecutive aligned frames before declaring "stable" (debounce).
    stable_frames: int = 5
    # How many consecutive missed frames before declaring the tin lost.
    lost_after: int = 10

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "CenteringConfig":
        """Return a config with defaults, overridden by any matching keys found
        in the YAML at `path` (default: config/centering_config.yaml next to this
        module). A missing file just yields plain defaults.

        Typically the file only holds target_x / target_y written by
        calibrate_sweet_spot.py, but any CenteringConfig field is accepted.
        `yaml` is imported lazily so the offline logic test can import this module
        without PyYAML installed (it always passes an explicit config)."""
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
    """
    Image-based centering monitor. Feed it DetectionResults every tick;
    it answers "is the tin where the arm can grasp it?" and, if not,
    suggests how the chassis should move. Stateless about hardware.
    """

    def __init__(self, config: Optional[CenteringConfig] = None):
        # No explicit config -> pick up the calibrated sweet spot from the YAML.
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

        # Smooth the detected center to reject single-frame detection jitter.
        cx, cy = float(best.center_x), float(best.center_y)
        if self._smoothed is None:
            self._smoothed = (cx, cy)
        else:
            a = cfg.ema_alpha
            self._smoothed = (a * self._smoothed[0] + (1 - a) * cx,
                              a * self._smoothed[1] + (1 - a) * cy)
        sx, sy = self._smoothed

        # Normalized error from the sweet spot.
        # error_x: + = tin right of target; error_y: + = tin closer to robot than target.
        error_x = cfg.mirror_x * (sx / detection.frame_width - cfg.target_x)
        error_y = cfg.image_y_is_backward * (sy / detection.frame_height - cfg.target_y)

        aligned = math.hypot(error_x, error_y) <= cfg.tolerance
        self._aligned_streak = self._aligned_streak + 1 if aligned else 0
        stable = self._aligned_streak >= cfg.stable_frames

        move = self._classify(error_x, error_y, aligned)

        # Continuous suggestion, same shape as ActionCommand.motion_vector
        # (forward, strafe, steer). Strafe is 0: differential chassis can't strafe.
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
        """Map the normalized error to a discrete suggestion. Each axis only
        contributes if its own component exceeds the (per-axis) tolerance,
        so a tiny lateral error doesn't cause needless turning."""
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
