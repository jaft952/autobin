"""
src/subsumption/layers/layer3_collect.py

Layer 3: Collect Litter — decides WHEN the tin is grabbable, halts the base,
and emits the grab as a semantic command. Execution (the actual servo
sequence) lives in ArmExecutor, after arbitration.

--- GRASP STRATEGY: arc grasp only ---
ArcGraspSolver.solve_with_band(nx, ny) interpolates a full hand-tuned
[CH1..CH5] pose from the calibrated arc grid. Every pose in that grid
physically worked on the real arm (sag/offsets baked in).
    -> arm_action='grab_arc', arm_params={'pose': [CH1..CH5]}

The model-IK fallback was removed on 2026-08-25 (source archived in
docs/ik_removed_2026-08-25.zip): it never fired, because it needed a
pixel_to_arm homography nobody had calibrated, and it aimed with a less
accurate solver than the grid it was backing up.

POSE ROUTING (v3): the segmentation mask classifies the tin as upright /
lying / axial (get_litter_pose). Upright tins use the upright arc grid;
lying tins use the separate LYING grid, with CH5 (wrist roll) computed from
the tin's floor angle. "axial" (seen end-on) grabs as straight-at-robot.

--- WHY THE BAND MATTERS ---
solve_with_band() also says WHY a None is None. BAND_TOO_FAR means keep
approaching, so this layer stands down and Layer 2 drives on. BAND_TOO_CLOSE
means the base has already overshot the nearest calibrated arc: standing
down there let Layer 2 keep closing in, which only made it worse, until
Layer 5 tripped and drove away from the tin. This layer now backs off
instead — the same recovery tests/test_ibvs_centering.py's live loop does.

--- WHY GRABBING WAITS ---
Camera inference mis-estimates position for a beat (motion blur, a partly
occluded mask), so ONE good frame is not evidence the tin is really in
position. Two gates, both mirrored from that same live loop:
    GRAB_STABLE_S      the whole trigger must hold continuously this long
    GRABBABLE_LATCH_S  one blank frame does not hand the tin back to Layer 2
While stabilizing, the layer stays active with a zero motion vector so the
base holds still and the reading can settle.

The solved position uses the litter's GROUND-CONTACT point (bbox bottom
center) for upright tins and the bbox CENTER for lying ones: that is what
each calibration grid was anchored to.

All solving here is pure math on calibration YAMLs (no hardware imports) —
Rule 2 holds.
"""
from __future__ import annotations

import time
from typing import Any, Optional

from src.subsumption.layers.base_layer import BaseLayer
from src.subsumption.arbitrator import ActionCommand
from src.arm.arc_grasp import ArcGraspSolver, BAND_TOO_CLOSE
from src.motion.calibration import MotionCalibration
import src.visual_servoing.reactive_controller as reactive_mod

# Front-ultrasonic confirmation before the arm fires. Keep in step with
GRAB_CONFIRM_CM = 35.0

# Continuous hold required before the grab fires, and how long a solved
# answer survives a blank frame. See "WHY GRABBING WAITS" above.
GRAB_STABLE_S = 1.0
GRABBABLE_LATCH_S = 2.0

# Overshot past the nearest calibrated arc: retreat in the same shape as
# tests/test_ibvs_centering.py's _retreat_pulse() -- a brief stop, then a
# bounded backward pulse at the same BACKUP_SPEED (reactive_controller.py),
# not an indefinite backoff at a separately-tuned fraction. Re-expressed as a
# non-blocking timed phase (RETREAT_GAP_S / RETREAT_PULSE_S) because a layer
# may never block or sleep -- the script's stop() -> sleep(0.05) ->
# apply(backward) -> sleep(0.3) -> stop() cycle becomes GAP then PULSE below.
RETREAT_GAP_S = 0.05
RETREAT_PULSE_S = 0.3
_CAL_FORWARD_SPEED = MotionCalibration().forward_speed  # duty -> fraction scale

# Layer 2 drives right up to the tick the arc grid first solves. Braking
# straight from that speed (MotionExecutor brakes, not coasts, for 'hold')
# jolts the chassis hard enough to knock the tin back out of the band that
# was just solved. Coast for this long first so momentum bleeds off under
# rolling friction, then brake to actually hold position -- same shape as
# the SETTLE_S halt-before-flip used in layer1_scan.py / layer4_emergency.py.
SETTLE_S = 0.15


class CollectLitterLayer(BaseLayer):
    """
    Layer 3: Collect Litter
    Priority: 3 (Medium)
    Behavior: When the detected tin sits inside the calibrated arc grid and
              the reading has settled, halts the base and commands an arc
              grab. Overshot past the nearest arc, it backs off instead.
    """

    def __init__(self, arc_solver: Optional[ArcGraspSolver] = None):
        super().__init__(layer_id=3)
        self.solver = arc_solver or ArcGraspSolver()
        self._ready_since: Optional[float] = None
        self._latched_until: float = 0.0
        self._retreating: bool = False   # True = mid-pulse, False = mid-gap
        self._retreat_until: float = 0.0
        self._was_too_close: bool = False
        self._suppressed_since: Optional[float] = None
        self._settle_until: float = 0.0
        self._prev_halted: bool = False   # were we already coasting/braking last tick?

    def reset(self) -> None:
        self._ready_since = None
        self._latched_until = 0.0
        self._retreating = False
        self._retreat_until = 0.0
        self._was_too_close = False
        self._suppressed_since = None
        self._settle_until = 0.0
        self._prev_halted = False

    # ── Arbitration ───────────────────────────────────────────────────────

    def notify_arbitration(self, won: bool) -> None:
        if not won and self._suppressed_since is None:
            self._suppressed_since = time.monotonic()

    def evaluate(self, sensors: Any) -> ActionCommand:
        now = time.monotonic()
        self._absorb_suppressed_time(now)
        solved, klass, point, band = self._solve(sensors)

        if band == BAND_TOO_CLOSE:
            self._ready_since = None
            self._latched_until = 0.0
            self._prev_halted = False
            return self._retreat_pulse(now)
        self._was_too_close = False

        if solved is None:
            self._ready_since = None
            if now < self._latched_until:
                # A blink, not a departure: hold the base rather than handing
                # the tin back to Layer 2 to be re-approached from scratch.
                return self._hold_or_settle(now, "GRAB HOLD (detection blinked)")
            self._prev_halted = False
            return ActionCommand(layer_id=self.layer_id, active=False)

        self._latched_until = now + GRABBABLE_LATCH_S

        if not self._ultrasonic_confirms(sensors):
            self._ready_since = None
            self._prev_halted = False
            return ActionCommand(layer_id=self.layer_id, active=False)

        if self._ready_since is None:
            self._ready_since = now
        stable_s = now - self._ready_since
        if stable_s < GRAB_STABLE_S:
            return self._hold_or_settle(
                now, f"GRAB HOLD (stabilizing {stable_s:.1f}s/{GRAB_STABLE_S:.0f}s)")

        nx, ny = point # type: ignore
        if klass == "upright":
            label = "upright"
        elif klass == "axial":
            label = "lying end-on"
        else:
            label = f"lying CH5={solved[4]:.0f}"
        self._prev_halted = True
        return ActionCommand(
            layer_id=self.layer_id,
            active=True,
            motion_vector=(0, 0, 0),          # halt base for the grab
            arm_action='grab_arc',
            # tin_pose picks the approach order (lying: elbow last)
            arm_params={'pose': solved, 'tin_pose': klass},
            message=f"ARC GRAB ({label}) @ nx={nx:.2f} ny={ny:.2f}",
        )

    def _absorb_suppressed_time(self, now: float) -> None:
        """Open-loop timers must not count time spent suppressed by Layer 5."""
        if self._suppressed_since is None:
            return
        paused = now - self._suppressed_since
        if self._ready_since is not None:
            self._ready_since += paused
        if self._latched_until:
            self._latched_until += paused
        if self._retreat_until:
            self._retreat_until += paused
        if self._settle_until:
            self._settle_until += paused
        self._suppressed_since = None

    def is_grabbable(self, sensors: Any) -> bool:
        """Can the arc grid solve this tin from where the robot stands?
        Pure — no timers touched, so Layer 5 and ArmExecutor can ask it
        without disturbing this layer's stability clock."""
        return self._solve(sensors)[0] is not None

    def can_still_in_grab_zone(self, sensors: Any) -> bool:
        """Re-run the grabbability check after a grab, once the arm is clear
        of the camera — catches a miss regardless of where a knocked-but-not-
        grabbed can ended up."""
        return self.is_grabbable(sensors)

    # ── Internals ─────────────────────────────────────────────────────────

    def _retreat_pulse(self, now: float) -> ActionCommand:
        """Bounded backward pulse with a brief stopped gap before it, same
        shape as test_ibvs_centering.py's _retreat_pulse() (stop -> backward
        at BACKUP_SPEED for RETREAT_PULSE_S -> stop), just re-expressed as a
        timed phase instead of blocking sleeps."""
        if not self._was_too_close:
            # Freshly entered BAND_TOO_CLOSE: always start with the stopped
            # gap, never straight into reverse (hardware_safety_patterns.md
            # rule 7 -- a direction flip needs a stop in between).
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

    def _hold_or_settle(self, now: float, message: str) -> ActionCommand:
        """First tick handing off from Layer 2's driving motion coasts instead
        of braking, so an instant brake at speed can't knock the tin out of
        the band the arc grid just solved. Already halted -> straight to hold."""
        if not self._prev_halted:
            self._settle_until = now + SETTLE_S
            self._prev_halted = True
        if now < self._settle_until:
            return ActionCommand(
                layer_id=self.layer_id, active=True,
                motion_vector=(0, 0, 0), arm_action='deploy',
                message="SETTLING before hold (coasting to a stop)",
            )
        return self._hold(message)

    def _hold(self, message: str) -> ActionCommand:
        """Base held still, arm parked at the travel pose. 'hold' brakes the
        wheels (see MotionExecutor._GRAB_ACTIONS) so the tin does not drift
        out of the band while the reading settles."""
        return ActionCommand(
            layer_id=self.layer_id,
            active=True,
            motion_vector=(0, 0, 0),
            arm_action='hold',
            message=message,
        )

    def _solve(self, sensors: Any):
        """(solved_or_None, klass, (nx, ny) or None, band)."""
        klass, angle = self._litter_pose(sensors)
        # Reference point differs per pose, and must match what the
        # calibration tool told the user to click:
        #   upright -> ground contact (bbox bottom-center): the tin meets the
        #              floor there, a stable anchor for distance.
        #   lying   -> bbox CENTER: the bottom edge drifts with orientation
        #              while the silhouette center tracks the graspable
        #              middle at every angle.
        if klass in ("lying", "axial"):
            pos = sensors.get_litter_position()
        else:
            pos = self._litter_point(sensors)
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
        """Front sensor agrees the tin is within arm range. No reading is NOT
        a veto: an off-center tin (a normal calibrated position) sits outside
        the sensor's narrow beam entirely, and vision plus the arc solver have
        already placed it. A real too-far reading still blocks."""
        getter = getattr(sensors, "get_obstacle_distance_cm", None)
        distance = getter() if getter is not None else None
        return distance is None or distance <= GRAB_CONFIRM_CM

    @staticmethod
    def _litter_point(sensors: Any) -> Optional[tuple]:
        """Ground-contact point if the sensor provides it, else bbox center."""
        getter = getattr(sensors, "get_litter_ground_contact", None)
        if getter is not None:
            pos = getter()
            if pos is not None:
                return pos
        return sensors.get_litter_position()

    @staticmethod
    def _litter_pose(sensors: Any) -> tuple:
        """(klass, angle_deg) from the segmentation mask, defaulting to
        ("upright", None) when the sensor has no pose information."""
        getter = getattr(sensors, "get_litter_pose", None)
        info = getter() if getter is not None else None
        if not info:
            return "upright", None
        return info.get("klass", "upright"), info.get("angle")
