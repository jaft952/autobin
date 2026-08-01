"""
src/subsumption/layers/layer3_collect.py

Layer 3: Collect Litter — decides WHEN the tin is grabbable and WITH WHICH
method, halts the base, and emits the grab as a semantic command. Execution
(the actual servo sequence) lives in ArmExecutor, after arbitration.

--- GRASP STRATEGY: arc grasp first, model IK as fallback ---
1. ARC GRASP (primary): ArcGraspSolver.solve(nx, ny) interpolates a full
   hand-tuned [CH1..CH5] pose from the calibrated arc grid. Every pose in
   that grid physically worked on the real arm (sag/offsets baked in), so
   inside the calibrated region this is the accurate path.
   -> arm_action='grab_arc', arm_params={'pose': [CH1..CH5]}

2. MODEL IK (fallback, ONLY outside the calibrated region): pixel ->
   homography (pixel_to_arm) -> floor point in meters -> ActionCommand
   carries the target; ArmExecutor runs GraspPlanner.ik_move() with sag
   compensation. Less accurate than the tuned poses (that's why IK was
   demoted from primary), but better than refusing to grab in areas nobody
   calibrated. The radial gate below keeps it from lunging at hopeless
   targets; GraspPlanner still refuses unreachable IK solutions honestly.
   -> arm_action='grab_ik', arm_params={'target_m': (x, y)}

POSE ROUTING (v3, 2026-07-08): the segmentation mask classifies the tin as
upright / lying / axial (get_litter_pose). Upright tins use the upright arc
grid (+ IK fallback). Lying tins use the separate LYING grid, with CH5
(wrist roll) computed from the tin's floor angle via the calibrated anchor
table in arc_grasp.yaml — no IK fallback for lying (that path was only
validated on standing tins).

If neither method applies, this layer stays INACTIVE — Layer 2 keeps
approaching until the tin enters somewhere we CAN grab. All solving here is
pure math on calibration YAMLs (no hardware imports), same hardware-free
status as ibvs_centering — Rule 2 holds.

The solved position uses the litter's GROUND-CONTACT point (bbox bottom
center) when the sensor provides it: both calibrations are anchored to where
the tin meets the floor, not the bbox center.
"""
from __future__ import annotations

import math
from typing import Any, Optional

from src.subsumption.layers.base_layer import BaseLayer
from src.subsumption.arbitrator import ActionCommand
from src.arm.arc_grasp import ArcGraspSolver
from src.arm.pixel_to_arm import PixelToArm

# Radial gate for the IK fallback (meters, arm frame, floor plane). Outside
# this annulus the arm physically can't make the grab, so don't halt the base
# for it — let Layer 2 keep driving closer. Tune against the real arm's reach
# (tip corrections were measured out to ~0.28 m horizontal reach).
IK_MIN_RADIUS_M = 0.12
IK_MAX_RADIUS_M = 0.32


class CollectLitterLayer(BaseLayer):
    """
    Layer 3: Collect Litter
    Priority: 3 (Medium)
    Behavior: When the detected tin is inside a grabbable region, halts the
              base and commands a grab — arc-grasp pose if the calibrated
              grid covers the spot, IK floor-target otherwise.
    """

    def __init__(self, arc_solver: Optional[ArcGraspSolver] = None,
                 pixel_to_arm: Optional[PixelToArm] = None):
        super().__init__(layer_id=3)
        self.solver = arc_solver or ArcGraspSolver()
        self.p2a = pixel_to_arm or PixelToArm()

    def evaluate(self, sensors: Any) -> ActionCommand:
        # Pose from segmentation: upright vs lying picks the calibration
        # grid; a lying tin's floor angle drives the wrist roll (CH5).
        # No pose info -> "upright" (the historical assumption).
        klass, angle = self._litter_pose(sensors)

        # REFERENCE POINT differs per pose, and must match what the
        # calibration tool told the user to click:
        #   upright -> ground contact (bbox bottom-center): the tin meets
        #              the floor there, stable anchor for distance.
        #   lying   -> bbox CENTER: the bottom edge drifts with orientation
        #              (pointing at the robot it's the near rim, half a can
        #              length in front of the grab point; sideways it's only
        #              a radius off), while the silhouette center tracks the
        #              graspable middle at every angle.
        if klass in ("lying", "axial"):
            pos = sensors.get_litter_position()          # bbox center
        else:
            pos = self._litter_point(sensors)            # ground contact
        if pos is None:
            return ActionCommand(layer_id=self.layer_id, active=False)
        nx, ny = pos

        if klass in ("lying", "axial"):
            # Lying tins: LYING grid only, no IK fallback — the IK path's
            # gripper-down waist grab was only ever validated on standing
            # tins; a lying can needs the calibrated poses + angle roll.
            # "axial" (seen end-on, mask is round) grabs as straight-at-robot.
            solved = self.solver.solve(nx, ny, pose="lying",
                                       angle_deg=None if klass == "axial" else angle)
            if solved is not None:
                ang_txt = "end-on" if klass == "axial" else f"{angle:.0f}deg"
                return ActionCommand(
                    layer_id=self.layer_id,
                    active=True,
                    motion_vector=(0, 0, 0),      # halt base for the grab
                    arm_action='grab_arc',
                    # tin_pose picks the approach order (lying: elbow last)
                    arm_params={'pose': solved, 'tin_pose': klass},
                    message=(f"ARC GRAB (lying {ang_txt}, CH5={solved[4]:.0f}) "
                             f"@ nx={nx:.2f} ny={ny:.2f}"),
                )
            return ActionCommand(layer_id=self.layer_id, active=False)

        # 1) Upright arc grasp: inside the calibrated grid this is proven.
        solved = self.solver.solve(nx, ny, pose="upright")
        if solved is not None:
            return ActionCommand(
                layer_id=self.layer_id,
                active=True,
                motion_vector=(0, 0, 0),          # halt base for the grab
                arm_action='grab_arc',
                arm_params={'pose': solved, 'tin_pose': 'upright'},
                message=f"ARC GRAB (upright) @ nx={nx:.2f} ny={ny:.2f}",
            )

        # 2) IK fallback: only for spots the upright grid doesn't cover.
        target = self._ik_target(nx, ny)
        if target is not None:
            x_m, y_m = target
            return ActionCommand(
                layer_id=self.layer_id,
                active=True,
                motion_vector=(0, 0, 0),
                arm_action='grab_ik',
                arm_params={'target_m': (x_m, y_m)},
                message=f"IK GRAB @ x={x_m:.3f}m y={y_m:.3f}m (outside arc grid)",
            )

        # Not grabbable from here — Layer 2 keeps approaching.
        return ActionCommand(layer_id=self.layer_id, active=False)

    # ── Internals ─────────────────────────────────────────────────────────

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

    def _ik_target(self, nx: float, ny: float) -> Optional[tuple]:
        """Normalized pixel -> floor point (m) via the homography, gated to
        the arm's physical annulus. None if uncalibrated or out of reach."""
        if not self.p2a.ready:
            return None
        res = self.p2a.cfg.get("resolution")
        if not res:
            return None                      # can't denormalize safely
        xy = self.p2a.transform(nx * float(res[0]), ny * float(res[1]))
        if xy is None:
            return None
        x_m, y_m = xy
        r = math.hypot(x_m, y_m)
        if y_m <= 0 or not (IK_MIN_RADIUS_M <= r <= IK_MAX_RADIUS_M):
            return None                      # behind the arm or outside reach
        return (x_m, y_m)
