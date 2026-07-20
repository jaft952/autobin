"""
src/visual_servoing/approach_planner.py

Metric approach planning for cruise mode — "drive to a point 31 cm in front of
the tin", not "react to the pixel error".

WHY THIS EXISTS
---------------
The pixel P-controller in _MotorWorker is a pure reaction: it turns the CURRENT
error into a CURRENT duty. YOLO on the Pi answers every ~100-300 ms, so the
duty the wheels are executing is always a decision about where the tin WAS.
While the base rolls, the tin gets closer, but the command doesn't shrink until
the next detection lands — so the base arrives at full approach speed and
pushes the tin away. Lowering the gain just makes it too slow to ever arrive.

The fix is not a better gain, it's a different signal. Here the vision frame
sets a GOAL IN METRES; between frames the base integrates the wheel duties it
is actually sending (dead reckoning) and shrinks the remaining distance itself.
The deceleration ramp is computed from that remaining distance, so it slows on
schedule even if the next detection never arrives. A detection is no longer a
command — it is a CORRECTION to an estimate the planner is already maintaining.

METRIC MODEL (GroundRange)
--------------------------
The tin stands on the floor, so its ground-contact pixel row v maps to a range
in metres by a 1-D projective function:

    y(v) = (a*v + b) / (c*v + 1)

Three parameters, fitted from the ruler samples already in
centering_config.yaml. This is the WELL-CONDITIONED half of the old
pixel_to_arm homography: refitting only the rows reproduces all six measured
distances to within 4 mm, while the full 8-DoF homography blows up because two
of the six lateral (x) measurements are inconsistent — one is even sign-flipped
(u=153, far LEFT of centre, recorded as x=+0.17). So range comes from the fit
and LATERAL comes from the pinhole relation instead:

    x = (u - cx) * y / f_px

f_px is a single focal length in pixels (~0.72 * frame_width for this camera's
FOV). Its exact value only scales the heading error, which is closed-loop
anyway — a 15% error in f_px costs a slightly hotter or lazier steer gain, not
a mis-aimed stop.

Above the horizon row (the pole of y(v)) range is meaningless, so the planner
reports "no metric estimate" and the caller falls back to plain pixel chasing —
which is fine two metres out, where centimetres don't matter.

DRIVE MODEL
-----------
Dead reckoning needs metres-per-second, and nothing in MotionCalibration is
metric (forward_speed=90.0 is a duty percent). So DriveModel holds v_max /
w_max at duty 1.0 and LEARNS them online: after each detection, compare the
distance the planner predicted it travelled against the distance the tin's
range actually dropped, and nudge the scale. Defaults are deliberately HIGH —
overestimating speed makes the planner think it has already covered more ground
than it has, so it stops SHORT. Underestimating is what rams the tin.

Learned values persist to config/drive_model.yaml, so the second run starts
calibrated.

FRAME CONVENTION (shared with set_motor_pwm / IBVS)
    x = right, y = forward, all metres, origin at the robot.
    bearing phi = atan2(x, y): + = the target is to the RIGHT.
    steer duty:  + = turn LEFT   (so steer = -k * phi)
    forward duty: + = forward
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Optional, Tuple

CONFIG_DIR = Path(__file__).parent / "config"
DRIVE_MODEL_PATH = CONFIG_DIR / "drive_model.yaml"
CENTERING_CONFIG_PATH = CONFIG_DIR / "centering_config.yaml"

# ── Ground model ─────────────────────────────────────────────────────────────
# Focal length as a fraction of frame width. 0.72 ≈ 70° horizontal FOV, right
# for the C270/Brio at 1280x720. Only scales the heading error (closed loop).
FOCAL_PX_PER_WIDTH = 0.72
# Rows closer than this to the horizon pole are refused: y(v) goes asymptotic
# there, so a 1-pixel wobble becomes metres.
HORIZON_MARGIN_PX = 45.0
# Beyond this the fit is extrapolating past every calibration sample; the
# planner declines and the caller chases pixels until the tin is nearer.
MAX_TRUSTED_RANGE_M = 1.30

# ── Drive model defaults (metres/sec and rad/sec at duty 1.0) ────────────────
# Biased HIGH on purpose — see module docstring. Learned down from here.
DEFAULT_V_MAX = 0.50
DEFAULT_W_MAX = 3.00
V_MAX_BOUNDS = (0.08, 1.20)
W_MAX_BOUNDS = (0.40, 8.00)
LEARN_GAIN = 0.15          # EMA weight per usable observation
LEARN_MIN_TRAVEL_M = 0.02  # ignore observations too small to be signal
LEARN_MIN_TURN_RAD = 0.09
LEARN_MAX_RATIO = 2.5      # a wilder mismatch than this is a bad detection

# ── Approach profile ─────────────────────────────────────────────────────────
DECEL_MPS2 = 0.35          # deceleration ramp: v = sqrt(2*a*d)
AIM_ENTER_RAD = math.radians(22)   # pivot in place above this heading error
AIM_EXIT_RAD = math.radians(9)
STEER_K = 1.6              # steer duty per radian of heading error while rolling
FINE_RANGE_M = 0.12        # inside this, stop cruising and take small steps
ARRIVE_RANGE_M = 0.035
ARRIVE_HEADING_RAD = math.radians(6)
# Fine phase: nudge, then hold still long enough for a fresh detection to
# arrive (inference + lag) before deciding again. Each nudge is sized to cover
# about half the remaining gap, so the approach converges geometrically in a
# handful of steps instead of creeping in fixed 1 cm increments.
PULSE_CLOSE_FRACTION = 0.5
PULSE_ON_MIN_S = 0.05
PULSE_ON_MAX_S = 0.25
PULSE_OFF_S = 0.35
# How long the planner may run on dead reckoning alone. The tin often leaves
# the frame bottom during the last few centimetres, exactly when stopping
# accurately matters most, so the plan outlives the detection.
COAST_S = 1.20
ODOM_HISTORY_S = 2.0
ODOM_SAMPLE_S = 0.01       # keep 10 ms odometry resolution, not 1 ms


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class GroundRange:
    """Ground-contact pixel -> (x, y) metres in the robot frame.

    Range from the fitted row model, lateral from the pinhole relation. See the
    module docstring for why this replaces the pixel_to_arm homography here.
    """

    def __init__(self, samples=None, focal_px: Optional[float] = None):
        self.a = self.b = self.c = None
        self.focal_px = focal_px
        self.n_samples = 0
        self.fit_rms_m: Optional[float] = None
        if samples is None:
            samples = self._load_samples()
        if samples:
            self._fit(samples)

    # ── construction ─────────────────────────────────────────────────────
    @staticmethod
    def _load_samples():
        """Ruler samples live with the pixel_to_arm calibration (its own file
        if it exists, else the legacy key in centering_config.yaml)."""
        import yaml
        for path, key in ((Path(__file__).resolve().parents[1] / "arm" / "config"
                           / "pixel_to_arm.yaml", None),
                          (CENTERING_CONFIG_PATH, "pixel_to_arm")):
            if not path.exists():
                continue
            data = yaml.safe_load(path.read_text()) or {}
            if key:
                data = data.get(key) or {}
            samples = data.get("samples")
            if samples:
                return samples
        return None

    def _fit(self, samples):
        """Least squares on y*(c*v + 1) = a*v + b, i.e. a*v + b - c*v*y = y."""
        import numpy as np
        v = np.array([float(s["v"]) for s in samples])
        y = np.array([float(s["y"]) for s in samples])
        if len(v) < 3:
            return
        A = np.c_[v, np.ones_like(v), -v * y]
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        a, b, c = (float(x) for x in sol)
        if c == 0:
            return
        self.a, self.b, self.c = a, b, c
        self.n_samples = len(v)
        pred = (a * v + b) / (c * v + 1.0)
        self.fit_rms_m = float(np.sqrt(((pred - y) ** 2).mean()))

    @property
    def ready(self) -> bool:
        return self.a is not None

    @property
    def horizon_v(self) -> float:
        """Image row where the fitted range goes to infinity."""
        return -1.0 / self.c if self.c else float("-inf")

    def status(self) -> str:
        if not self.ready:
            return "ground range: NOT calibrated (no ruler samples) — pixel mode only"
        return (f"ground range: fitted from {self.n_samples} samples, "
                f"rms {self.fit_rms_m * 100:.1f} cm, horizon row "
                f"{self.horizon_v:.0f}, trusted to {MAX_TRUSTED_RANGE_M:.2f} m")

    # ── runtime ──────────────────────────────────────────────────────────
    def range_m(self, v: float) -> Optional[float]:
        """Range in metres for a ground-contact row, or None if untrustworthy."""
        if not self.ready:
            return None
        if v <= self.horizon_v + HORIZON_MARGIN_PX:
            return None                      # at/above the horizon: meaningless
        y = (self.a * v + self.b) / (self.c * v + 1.0)
        if y <= 0.02 or y > MAX_TRUSTED_RANGE_M:
            return None
        return y

    def project(self, u: float, v: float, frame_w: float) -> Optional[Tuple[float, float]]:
        """Ground-contact pixel -> (x right, y forward) in metres, or None."""
        y = self.range_m(v)
        if y is None:
            return None
        f = self.focal_px or (FOCAL_PX_PER_WIDTH * frame_w)
        x = (u - frame_w * 0.5) * y / f
        return (x, y)


class DriveModel:
    """Duty <-> velocity, learned online and persisted."""

    def __init__(self, path: Path = DRIVE_MODEL_PATH):
        self.path = Path(path)
        self.v_max = DEFAULT_V_MAX
        self.w_max = DEFAULT_W_MAX
        self.updates = 0
        self._dirty = False
        self.load()

    def load(self):
        try:
            import yaml
            if self.path.exists():
                data = yaml.safe_load(self.path.read_text()) or {}
                self.v_max = _clamp(float(data.get("v_max_mps", self.v_max)), *V_MAX_BOUNDS)
                self.w_max = _clamp(float(data.get("w_max_radps", self.w_max)), *W_MAX_BOUNDS)
                self.updates = int(data.get("updates", 0))
        except Exception as exc:
            print(f"[drive_model] using defaults ({exc})")

    def save(self):
        if not self._dirty:
            return
        try:
            import yaml
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(yaml.safe_dump({
                "v_max_mps": round(self.v_max, 4),
                "w_max_radps": round(self.w_max, 4),
                "updates": self.updates,
                "note": "learned by ApproachPlanner; delete to reset to defaults",
            }, sort_keys=False))
            self._dirty = False
        except Exception as exc:
            print(f"[drive_model] save failed ({exc})")

    def status(self) -> str:
        return (f"drive model: v_max {self.v_max:.3f} m/s, w_max {self.w_max:.2f} rad/s "
                f"at duty 1.0 ({self.updates} learned updates)")

    # predicted -> actual comparison; ratio > 1 means we move FASTER than modelled
    def _learn(self, attr: str, predicted: float, actual: float, bounds):
        if abs(predicted) < 1e-6:
            return
        ratio = actual / predicted
        if not (1.0 / LEARN_MAX_RATIO) < ratio < LEARN_MAX_RATIO:
            return                       # bad detection, not a bad model
        cur = getattr(self, attr)
        setattr(self, attr, _clamp(cur * (1 - LEARN_GAIN) + cur * ratio * LEARN_GAIN, *bounds))
        self.updates += 1
        self._dirty = True

    def learn_linear(self, predicted_m: float, actual_m: float):
        if abs(predicted_m) >= LEARN_MIN_TRAVEL_M:
            self._learn("v_max", predicted_m, actual_m, V_MAX_BOUNDS)

    def learn_yaw(self, predicted_rad: float, actual_rad: float):
        if abs(predicted_rad) >= LEARN_MIN_TURN_RAD:
            self._learn("w_max", predicted_rad, actual_rad, W_MAX_BOUNDS)


@dataclass
class ApproachCommand:
    forward: float           # final duty to send, already scaled
    steer: float             # final duty to send, already scaled
    phase: str               # aim | drive | fine | done | stale
    distance_m: float
    heading_rad: float

    @property
    def moving(self) -> bool:
        return abs(self.forward) > 0.005 or abs(self.steer) > 0.005


class ApproachPlanner:
    """Holds the waypoint in metres and drives the base onto it.

    Waypoint W = (right, forward) metres the ROBOT must travel so the tin ends
    up on the calibrated grasp spot. observe() corrects it from vision (with the
    vision latency taken out); predict() advances it from the duties actually
    sent; command() turns the remaining W into duties.
    """

    def __init__(self, drive_model: Optional[DriveModel] = None,
                 max_forward_duty: float = 1.0, max_steer_duty: float = 1.0,
                 stall_duty: float = 0.22):
        self.model = drive_model or DriveModel()
        self.max_forward = max(0.05, max_forward_duty)
        self.max_steer = max(0.05, max_steer_duty)
        self.stall = max(0.02, stall_duty)

        self.W: Optional[Tuple[float, float]] = None
        self.last_obs_t = 0.0
        self._aiming = False
        self._pulse_until = 0.0
        self._pulse_on = False
        # Cumulative odometry (t, forward metres, left-positive radians) so an
        # observation can be replayed forward from its capture time.
        self._odom: Deque[Tuple[float, float, float]] = deque()
        self._s_cum = 0.0
        self._psi_cum = 0.0
        self._odom.append((0.0, 0.0, 0.0))
        # Between-observation bookkeeping for the online drive-model fit.
        self._pred_at_last_obs: Optional[Tuple[float, float, Tuple[float, float]]] = None

    # ── state ────────────────────────────────────────────────────────────
    def reset(self):
        self.W = None
        self._aiming = False
        self._pulse_on = False
        self._pulse_until = 0.0
        self._pred_at_last_obs = None

    @property
    def has_plan(self) -> bool:
        return self.W is not None

    def distance(self) -> float:
        return math.hypot(*self.W) if self.W else 0.0

    def heading(self) -> float:
        return math.atan2(self.W[0], self.W[1]) if self.W else 0.0

    # ── prediction (motor thread, ~1 kHz) ────────────────────────────────
    def predict(self, dt: float, forward_duty: float, steer_duty: float, now: float):
        """Advance the waypoint by the motion the wheels just performed."""
        if dt <= 0:
            return
        s = self.model.v_max * forward_duty * dt        # metres forward
        psi = self.model.w_max * steer_duty * dt        # radians, + = left
        self._s_cum += s
        self._psi_cum += psi
        # Every tick accumulates, but only every ODOM_SAMPLE_S is kept: the
        # history is scanned per detection, and 1 kHz entries would make that
        # scan thousands of items long for no extra resolution.
        if now - self._odom[-1][0] >= ODOM_SAMPLE_S:
            self._odom.append((now, self._s_cum, self._psi_cum))
            while len(self._odom) > 2 and now - self._odom[0][0] > ODOM_HISTORY_S:
                self._odom.popleft()
        if self.W is not None:
            self.W = self._advance(self.W, s, psi)

    @staticmethod
    def _advance(W, s: float, psi: float):
        """Waypoint in the NEW robot frame after driving s forward and turning
        psi left. Rotating left moves the target further to the right."""
        x, y = W[0], W[1] - s
        c, sn = math.cos(psi), math.sin(psi)
        return (x * c + y * sn, y * c - x * sn)

    def _odom_at(self, t: float) -> Tuple[float, float]:
        """Cumulative (metres, radians) at time t — the newest entry at or
        before t, so a measurement can be replayed from when it was captured."""
        best = self._odom[0]
        for entry in self._odom:
            if entry[0] > t:
                break
            best = entry
        return best[1], best[2]

    # ── correction (vision thread rate) ──────────────────────────────────
    def observe(self, offset_xy: Tuple[float, float], capture_t: float, now: float):
        """Correct the plan from a metric measurement.

        offset_xy: (right, forward) metres the robot must travel, as measured
                   in the frame captured at capture_t.
        The measurement is aged: replay the odometry from capture_t to now so
        the plan describes where the tin is NOW, not one inference ago. This is
        the whole point of the module — without it the base acts on a pose it
        left behind two tenths of a second ago.
        """
        s_then, psi_then = self._odom_at(capture_t)
        W_now = self._advance(offset_xy,
                              self._s_cum - s_then,
                              self._psi_cum - psi_then)

        # Online drive-model fit: the plan thought it travelled pred_s since
        # the last observation; the tin's measured range says it travelled
        # (W_prev.y - W_now.y). The ratio corrects v_max. Each axis is learned
        # only from motion that ISOLATES it — a range drop is clean forward
        # travel only if we were barely turning, and a bearing change is clean
        # yaw only if we were barely translating (driving straight past an
        # off-axis tin swings its bearing too).
        if self._pred_at_last_obs is not None:
            s_prev, psi_prev, W_prev = self._pred_at_last_obs
            pred_s = self._s_cum - s_prev
            pred_psi = self._psi_cum - psi_prev
            if abs(pred_psi) < LEARN_MIN_TURN_RAD:
                self.model.learn_linear(pred_s, W_prev[1] - W_now[1])
            if abs(pred_s) < LEARN_MIN_TRAVEL_M:
                self.model.learn_yaw(pred_psi,
                                     math.atan2(W_now[0], W_now[1])
                                     - math.atan2(W_prev[0], W_prev[1]))

        self.W = W_now
        self.last_obs_t = now
        self._pred_at_last_obs = (self._s_cum, self._psi_cum, W_now)

    # ── command (motor thread) ───────────────────────────────────────────
    def _pulse_length(self, d: float, phi: float) -> float:
        """How long the next nudge should run: enough to close about half the
        remaining error at stall duty, bounded so one bad estimate can't turn
        into a lunge."""
        if abs(phi) > ARRIVE_HEADING_RAD:
            want = abs(phi) * PULSE_CLOSE_FRACTION / max(
                self.model.w_max * self.stall, 1e-6)
        else:
            want = (d - ARRIVE_RANGE_M) * PULSE_CLOSE_FRACTION / max(
                self.model.v_max * self.stall, 1e-6)
        return _clamp(want, PULSE_ON_MIN_S, PULSE_ON_MAX_S)

    def command(self, now: float) -> ApproachCommand:
        if self.W is None:
            return ApproachCommand(0.0, 0.0, "stale", 0.0, 0.0)
        d, phi = self.distance(), self.heading()
        if now - self.last_obs_t > COAST_S:
            # Ran out of vision AND out of plan — stop rather than guess.
            return ApproachCommand(0.0, 0.0, "stale", d, phi)

        if d <= ARRIVE_RANGE_M and abs(phi) <= ARRIVE_HEADING_RAD:
            return ApproachCommand(0.0, 0.0, "done", d, phi)

        # Pivot in place when badly off-heading. Hysteresis stops the state
        # chattering at the boundary; a pivot ALSO leaves the arrival distance
        # untouched, which is why it is worth doing separately from the drive.
        if self._aiming:
            self._aiming = abs(phi) > AIM_EXIT_RAD
        elif abs(phi) > AIM_ENTER_RAD:
            self._aiming = True
        if self._aiming:
            steer = -math.copysign(
                _clamp(abs(phi) * STEER_K, self.stall, self.max_steer), phi)
            return ApproachCommand(0.0, steer, "aim", d, phi)

        # Final centimetres: discrete nudges, holding still between them so a
        # fresh detection lands before the next one. Rolling continuously here
        # is what nudges the tin out of the grasp zone.
        if d <= FINE_RANGE_M:
            if now >= self._pulse_until:
                self._pulse_on = not self._pulse_on
                self._pulse_until = now + (self._pulse_length(d, phi)
                                           if self._pulse_on else PULSE_OFF_S)
            if not self._pulse_on:
                return ApproachCommand(0.0, 0.0, "fine", d, phi)
            # Turn first, then close the gap — same "aim, then advance" order
            # the discrete IBVS moves use, since a nudge at the wrong heading
            # costs two corrections.
            if abs(phi) > ARRIVE_HEADING_RAD:
                return ApproachCommand(0.0, -math.copysign(self.stall, phi),
                                       "fine", d, phi)
            fwd = self.stall if d > ARRIVE_RANGE_M else 0.0
            return ApproachCommand(fwd, 0.0, "fine", d, phi)

        # Cruise: speed from the deceleration ramp on the REMAINING distance,
        # which the dead reckoning keeps current between detections. This is
        # what makes the base slow down on approach without waiting to be told.
        v_des = math.sqrt(2.0 * DECEL_MPS2 * max(0.0, d - ARRIVE_RANGE_M))
        duty = _clamp(v_des / max(self.model.v_max, 1e-6), self.stall, self.max_forward)
        steer = _clamp(-phi * STEER_K, -self.max_steer, self.max_steer)
        return ApproachCommand(duty, steer, "drive", d, phi)
