from __future__ import annotations

import time
from typing import Any, Optional

from src.subsumption.layers.base_layer import BaseLayer
from src.subsumption.arbitrator import ActionCommand
from src.hardware.sensors.interfaces import LitterSnapshot
from src.arm.arc_grasp import ArcGraspSolver, BAND_TOO_CLOSE
from src.motion.calibration import MotionCalibration
import src.visual_servoing.reactive_controller as reactive_mod

GRAB_CONFIRM_CM = 35.0
GRAB_STABLE_S = 2.0
GRABBABLE_LATCH_S = 2.0

# Max nx/ny swing allowed while still counting the tin as stopped.
POSE_STABLE_EPS = 0.03


RETREAT_GAP_S = 0.05
RETREAT_PULSE_S = 0.3
_CAL_FORWARD_SPEED = MotionCalibration().forward_speed  # duty -> fraction scale


class CollectLitterLayer(BaseLayer):
    """Layer 3: halts base and grabs a settled, in-range tin; backs off if overshot."""

    def __init__(self, arc_solver: Optional[ArcGraspSolver] = None):
        super().__init__(layer_id=3)
        self.solver = arc_solver or ArcGraspSolver()
        self._ready_since: Optional[float] = None
        self._latched_until: float = 0.0
        self._retreating: bool = False   # True = mid-pulse, False = mid-gap
        self._retreat_until: float = 0.0
        self._was_too_close: bool = False
        self._suppressed_since: Optional[float] = None
        self._points: list = []          # (t, nx, ny) samples

    def reset(self) -> None:
        self._ready_since = None
        self._latched_until = 0.0
        self._retreating = False
        self._retreat_until = 0.0
        self._was_too_close = False
        self._suppressed_since = None
        self._points.clear()

    # ── Arbitration ───────────────────────────────────────────────────────

    def notify_arbitration(self, won: bool) -> None:
        if not won and self._suppressed_since is None:
            self._suppressed_since = time.monotonic()

    def evaluate(self, sensors: Any) -> ActionCommand:
        now = time.monotonic()
        self._absorb_suppressed_time(now)
        solved, klass, point, band = self._solve(sensors)

        if band == BAND_TOO_CLOSE:
            self._restart_stability()
            self._latched_until = 0.0
            return self._retreat_pulse(now)
        self._was_too_close = False

        if solved is None:
            self._restart_stability()
            if now < self._latched_until:
                # Blink, not a departure: hold instead of handing back to Layer 2.
                return self._hold("GRAB HOLD (detection blinked)")
            return ActionCommand(layer_id=self.layer_id, active=False)

        self._latched_until = now + GRABBABLE_LATCH_S

        if not self._ultrasonic_confirms(sensors):
            # Hold, don't stand down: one noisy ping must not hand control back to Layer 2.
            self._restart_stability()
            return self._hold("GRAB HOLD (ultrasonic not yet confirming range)")

        nx, ny = point # type: ignore
        self._track_point(now, nx, ny)

        if self._ready_since is None:
            self._ready_since = now
        stable_s = now - self._ready_since
        if stable_s < GRAB_STABLE_S:
            return self._hold(f"GRAB HOLD (stabilizing {stable_s:.1f}s"
                              f"/{GRAB_STABLE_S:.0f}s)")

        drift = self._point_drift()
        if drift > POSE_STABLE_EPS:
            return self._hold(f"GRAB HOLD (target still moving, "
                              f"drift {drift:.3f}/{POSE_STABLE_EPS})")

        if klass == "upright":
            label = "upright"
        elif klass == "axial":
            label = "lying end-on"
        else:
            label = f"lying CH5={solved[4]:.0f}"
        return ActionCommand(
            layer_id=self.layer_id,
            active=True,
            motion_vector=(0, 0, 0),          # halt base for the grab
            arm_action='grab_arc',
            arm_params={'pose': solved, 'tin_pose': klass},  # tin_pose sets approach order
            message=f"ARC GRAB ({label}) @ nx={nx:.2f} ny={ny:.2f}",
        )

    def _absorb_suppressed_time(self, now: float) -> None:
        """Don't count time suppressed by Layer 5 in the open-loop timers."""
        if self._suppressed_since is None:
            return
        paused = now - self._suppressed_since
        if self._ready_since is not None:
            self._ready_since += paused
        if self._latched_until:
            self._latched_until += paused
        if self._retreat_until:
            self._retreat_until += paused
        self._points.clear()
        self._suppressed_since = None

    def is_grabbable(self, sensors: Any) -> bool:
        """Can the arc grid solve this tin now? Pure, no timers touched."""
        return self._solve(sensors)[0] is not None

    def can_still_in_grab_zone(self, sensors: Any) -> bool:
        """Re-check grabbability after a grab, once the arm clears the camera."""
        return self.is_grabbable(sensors)

    # ── Internals ─────────────────────────────────────────────────────────

    def _retreat_pulse(self, now: float) -> ActionCommand:
        """Bounded backward pulse: stop -> backward -> stop, as timed phases."""
        if not self._was_too_close:
            # Always stop before reversing, never flip direction directly.
            self._was_too_close = True
            self._retreating = False
            self._retreat_until = now + RETREAT_GAP_S
        elif now >= self._retreat_until:
            self._retreating = not self._retreating
            self._retreat_until = now + (RETREAT_PULSE_S if self._retreating
                                         else RETREAT_GAP_S)

        if self._retreating:
            v_x = -reactive_mod.BACKUP_SPEED / _CAL_FORWARD_SPEED
            return ActionCommand(
                layer_id=self.layer_id, active=True,
                motion_vector=(v_x, 0, 0), arm_action='deploy',
                message="OVERSHOT past nearest arc - retreat pulse",
            )
        return ActionCommand(
            layer_id=self.layer_id, active=True,
            motion_vector=(0, 0, 0), arm_action='deploy',
            message="OVERSHOT past nearest arc - pulse gap",
        )

    def _restart_stability(self) -> None:
        """Abandon the settling window; old timing and points no longer apply."""
        self._ready_since = None
        self._points.clear()

    def _track_point(self, now: float, nx: float, ny: float) -> None:
        self._points.append((now, nx, ny))
        cutoff = now - GRAB_STABLE_S
        self._points = [p for p in self._points if p[0] >= cutoff]

    def _point_drift(self) -> float:
        """Widest nx/ny swing in the stable window; 0.0 until 2+ samples."""
        if len(self._points) < 2:
            return 0.0
        xs = [p[1] for p in self._points]
        ys = [p[2] for p in self._points]
        return max(max(xs) - min(xs), max(ys) - min(ys))

    def _hold(self, message: str) -> ActionCommand:
        """Base held still (braked) so the tin doesn't drift while settling."""
        return ActionCommand(
            layer_id=self.layer_id,
            active=True,
            motion_vector=(0, 0, 0),
            arm_action='hold',
            message=message,
        )

    def _solve(self, sensors: Any):
        """(solved_or_None, klass, (nx, ny) or None, band)."""
        snap = self._snapshot(sensors)
        klass = snap.klass or "upright"
        angle = snap.angle
        # Reference point per pose: upright uses ground contact, lying uses bbox center.
        if klass in ("lying", "axial"):
            pos = snap.center
        else:
            pos = snap.ground_contact or snap.center
        if pos is None:
            return None, klass, None, None

        nx, ny = pos
        if klass in ("lying", "axial"):
            solved, band = self.solver.solve_with_band(
                nx, ny, pose="lying",
                angle_deg=None if klass == "axial" else angle)
        else:
            solved, band = self.solver.solve_with_band(nx, ny, pose="upright")
        return solved, klass, (nx, ny), band

    @staticmethod
    def _ultrasonic_confirms(sensors: Any) -> bool:
        """Front sensor agrees tin is in range. No reading is not a veto; too far still blocks."""
        getter = getattr(sensors, "get_obstacle_distance_cm", None)
        distance = getter() if getter is not None else None
        return distance is None or distance <= GRAB_CONFIRM_CM

    @staticmethod
    def _snapshot(sensors: Any) -> LitterSnapshot:
        """Litter facts from one frame; falls back to per-getter for older sensors."""
        getter = getattr(sensors, "snapshot", None)
        if getter is not None:
            snap = getter()
            if snap is not None:
                return snap

        pose_getter = getattr(sensors, "get_litter_pose", None)
        info = pose_getter() if pose_getter is not None else None
        contact_getter = getattr(sensors, "get_litter_ground_contact", None)
        return LitterSnapshot(
            center=sensors.get_litter_position(),
            ground_contact=contact_getter() if contact_getter is not None else None,
            klass=info.get("klass") if info else None,
            angle=info.get("angle") if info else None,
        )
