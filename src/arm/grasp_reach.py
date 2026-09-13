"""Checks where the can is compared with the arm's reach."""
from __future__ import annotations

from typing import Any, NamedTuple, Optional


class GraspReach(NamedTuple):
    pose: Optional[list]
    klass: str
    point: Optional[tuple]
    band: Optional[str]


def solve_reach(solver, sensors: Any) -> GraspReach:
    """Solve the grasp pose for the can from the robot's current spot."""
    klass, angle = _litter_pose(sensors)
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
    """Point where the can touches the floor, or the box centre."""
    getter = getattr(sensors, "get_litter_ground_contact", None)
    if getter is not None:
        pos = getter()
        if pos is not None:
            return pos
    return sensors.get_litter_position()


def _litter_pose(sensors: Any) -> tuple:
    """Can pose and angle from the mask, upright if unknown."""
    getter = getattr(sensors, "get_litter_pose", None)
    info = getter() if getter is not None else None
    if not info:
        return "upright", None
    return info.get("klass", "upright"), info.get("angle")
