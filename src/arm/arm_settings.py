"""Single source of truth for arm poses, gripper angles and move pacing.

The autonomous planner (grasp_planner) and the manual tools in tests/ both
read these, so retuning a value here changes every path. Tools tune at
runtime through MotionProfile's setters instead of redeclaring constants.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.hardware.actuators.pca9685_driver import clamp_channel_angle

HOME_ANGLES     = [100.0, 30.0, 0.0, 150.0, 90.0]
BIN_DROP_ANGLES = [96.7, 96.7, 100.0, 20.0, 90.0]
GRAB_ANGLES     = [101.0, 106.0, 40.0, 180.0, 80.0]
ARC_LIFT_ARM    = [96.7, 96.7, 100.0, 100.0, 90.0]

GRIPPER_OPEN = 50.0
GRIPPER_CLOSED = 4.0

# Grab order per tin pose: the LAST channel is the one that descends onto the
# tin — upright: CH2 shoulder, lying: CH3 elbow.
ARC_GRAB_ORDER = {
    "upright": [0, 4, 2, 3, 1],
    "lying":   [0, 4, 3, 1, 2],
}

# Move pacing: step_deg per step_delay seconds is read as a SPEED by
# stepped_move. Gentle = lower peak current on a weak supply.
ARC_STEP_DEG   = 5.0
ARC_STEP_DELAY = 0.15
JOG_STEP_DELAY = 0.5

CH_NAMES = ["CH1 base", "CH2 shoulder", "CH3 elbow", "CH4 wrist",
            "CH5 roll", "CH6 grip"]

def grab_order(tin_pose: str):
    """Channel order for a grab; 'axial' tins are grabbed like lying ones."""
    return ARC_GRAB_ORDER["lying" if tin_pose in ("lying", "axial") else "upright"]


@dataclass
class MotionProfile:
    """Move pacing + gripper angles, tunable at runtime by the manual tools."""

    step_deg: float = ARC_STEP_DEG
    step_delay: float = ARC_STEP_DELAY
    gripper_open: float = GRIPPER_OPEN
    gripper_closed: float = GRIPPER_CLOSED

    def set_speed(self, step_deg: Optional[float] = None,
                  step_delay: Optional[float] = None) -> "MotionProfile":
        if step_deg is not None:
            self.step_deg = max(0.1, float(step_deg))
        if step_delay is not None:
            self.step_delay = max(1e-3, float(step_delay))
        return self

    def set_gripper_angles(self, open_deg: Optional[float] = None,
                           closed_deg: Optional[float] = None) -> "MotionProfile":
        # Clamped through the driver's safe window so no tool can set an angle
        # the hardware would stall against.
        if open_deg is not None:
            self.gripper_open = clamp_channel_angle(5, open_deg)
        if closed_deg is not None:
            self.gripper_closed = clamp_channel_angle(5, closed_deg)
        return self

    @property
    def speed_dps(self) -> float:
        return self.step_deg / max(self.step_delay, 1e-6)


def jog_profile() -> MotionProfile:
    """Profile for hand-jogging tools: grab pacing, but slower."""
    return MotionProfile(step_delay=JOG_STEP_DELAY)
