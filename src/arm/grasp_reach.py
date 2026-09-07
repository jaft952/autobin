"""
src/arm/grasp_reach.py

Where the tin sits relative to the arm's calibrated reach.

The layer that DRIVES the base and the layer that GRABS both need this
answer, and they must not disagree: Layer 2 used to stop on its own
monocular distance estimate while Layer 3 judged reach off the arc grid,
so the base parked at 25 cm on one measurement while the grid called the
same spot an overshoot -- approach, back off, approach, back off.

Kept as a free function over a solver, so each layer owns its own
ArcGraspSolver and neither has to import the other.
"""
from __future__ import annotations

from typing import Any, NamedTuple, Optional


class GraspReach(NamedTuple):
    pose: Optional[list]      # [CH1..CH5] the arm would use, None if unreachable
    klass: str                # 'upright' | 'lying' | 'axial'
    point: Optional[tuple]    # (nx, ny) the solve was anchored on
    band: Optional[str]       # BAND_* from src.arm.arc_grasp, None if no point


def solve_reach(solver, sensors: Any) -> GraspReach:
    """Solve the tin's grasp pose from where the robot stands right now.

    Pure: no state anywhere, so a layer can ask on every tick without
    disturbing its own timers.
    """
    klass, angle = _litter_pose(sensors)
    # Reference point differs per pose, and must match what the calibration
    # tool told the user to click:
    #   upright -> ground contact (bbox bottom-center): the tin meets the
    #              floor there, a stable anchor for distance.
    #   lying   -> bbox CENTER: the bottom edge drifts with orientation while
    #              the silhouette center tracks the graspable middle at every
    #              angle.
    lying = klass in ("lying", "axial")
    pos = sensors.get_litter_position() if lying else _litter_point(sensors)
    if pos is None:
        return GraspReach(None, klass, None, None)

    nx, ny = pos
    if lying:
        pose, band = solver.solve_with_band(
            nx, ny, pose="lying",
            angle_deg=None if klass == "axial" else angle)
    else:
        pose, band = solver.solve_with_band(nx, ny, pose="upright")
    return GraspReach(pose, klass, (nx, ny), band)


def _litter_point(sensors: Any) -> Optional[tuple]:
    """Ground-contact point if the sensor provides it, else bbox center."""
    getter = getattr(sensors, "get_litter_ground_contact", None)
    if getter is not None:
        pos = getter()
        if pos is not None:
            return pos
    return sensors.get_litter_position()


def _litter_pose(sensors: Any) -> tuple:
    """(klass, angle_deg) from the segmentation mask, defaulting to
    ("upright", None) when the sensor has no pose information."""
    getter = getattr(sensors, "get_litter_pose", None)
    info = getter() if getter is not None else None
    if not info:
        return "upright", None
    return info.get("klass", "upright"), info.get("angle")
