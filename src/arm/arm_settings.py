"""Arm poses, gripper angles and move speed settings."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.hardware.actuators.pca9685_driver import clamp_channel_angle

HOME_ANGLES     = [100.0, 100.0, 85.0, 0.0, 80.0]
BIN_DROP_ANGLES = [100.0, 100.0, 85.0, 10.0, 80.0]
GRAB_ANGLES     = [101.0, 106.0, 40.0, 180.0, 80.0]
ARC_LIFT_ARM    = [96.7, 96.7, 100.0, 100.0, 80.0]

GRIPPER_OPEN = 50.0
GRIPPER_CLOSED = 2.0

ARC_GRAB_ORDER = {
    "upright": [0, 4, 2, 3, 1],
    "lying":   [0, 4, 3, 1, 2],
}

ARC_STEP_DEG   = 5.0
ARC_STEP_DELAY = 0.15
JOG_STEP_DELAY = 0.5

SERVO_NEUTRAL_CMD = [0.0, 96.7, 96.7, 100.0, 100.0, 90.0, 80.0]

CH_NAMES = ["CH1 base", "CH2 shoulder", "CH3 elbow", "CH4 wrist",
            "CH5 roll", "CH6 grip"]

def grab_order(tin_pose: str):
    """Order to move the channels in during a grab."""
    return ARC_GRAB_ORDER["lying" if tin_pose in ("lying", "axial") else "upright"]


@dataclass
class MotionProfile:
    """Move speed and gripper angles that can be changed at run time."""

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
        if open_deg is not None:
            self.gripper_open = clamp_channel_angle(5, open_deg)
        if closed_deg is not None:
            self.gripper_closed = clamp_channel_angle(5, closed_deg)
        return self

    @property
    def speed_dps(self) -> float:
        return self.step_deg / max(self.step_delay, 1e-6)


def jog_profile() -> MotionProfile:
    """Slower move settings for manual jogging tools."""
    return MotionProfile(step_delay=JOG_STEP_DELAY)
