"""Arm driver: tracks the pose, moves smoothly and runs the grasp."""
from __future__ import annotations

from typing import Optional

from src.arm.arm_settings import (
    ARC_LIFT_ARM, BIN_DROP_ANGLES, CH_NAMES, GRAB_ANGLES, HOME_ANGLES,
    MotionProfile, grab_order,
)
from src.hardware.actuators.pca9685_driver import (
    ArmActuator, clamp_channel_angle, load_last_pose, save_last_pose,
    stepped_move,
)

class GraspPlanner:
    """Sends solved arm poses to the servo driver."""

    def __init__(self, actuator: Optional[ArmActuator] = None,
                 profile: Optional[MotionProfile] = None,
                 release_on_start: bool = False):
        """Set up the arm. release_on_start turns the servos off at once."""
        self.actuator = actuator or ArmActuator()
        self.profile = profile or MotionProfile()
        self.poses = {
            "home": list(HOME_ANGLES),
            "lift": list(ARC_LIFT_ARM),
            "bin":  list(BIN_DROP_ANGLES),
            "grab": list(GRAB_ANGLES),
        }
        last = load_last_pose()
        if last is not None:
            self.arm, self.gripper = last[:5], last[5]
            print(f"[arm] resuming last commanded pose "
                  f"{[round(v, 1) for v in self.arm]} grip={self.gripper:.1f}")
        else:
            self.arm = list(self.poses["home"])
            self.gripper = self.profile.gripper_open
            print("[arm] no saved pose — assuming HOME; call home() if the arm "
                  "isn't actually there.")
        if release_on_start:
            self.release()

    def set_speed(self, step_deg=None, step_delay=None) -> MotionProfile:
        """Change how fast the arm moves."""
        return self.profile.set_speed(step_deg, step_delay)

    def set_gripper_angles(self, open_deg=None, closed_deg=None) -> MotionProfile:
        """Change the gripper open and close angles."""
        return self.profile.set_gripper_angles(open_deg, closed_deg)

    def move_channel(self, ch: int, value: float) -> None:
        """Move one channel smoothly to an angle."""
        start = list(self.arm) + [self.gripper]
        target = list(start)
        target[ch] = clamp_channel_angle(ch, float(value))
        stepped_move(self.actuator, start, target,
                     self.profile.step_deg, self.profile.step_delay)
        self.actuator.set_channel_angle(ch, target[ch])
        if ch < 5:
            self.arm[ch] = target[ch]
        else:
            self.gripper = target[ch]
        save_last_pose(list(self.arm) + [self.gripper])

    def goto(self, target, label: str = "") -> None:
        """Move CH1 to CH5 to a pose, one channel at a time."""
        if isinstance(target, str):
            if target not in self.poses:
                raise ValueError(f"unknown pose '{target}'; "
                                 f"known: {list(self.poses)}")
            label = label or f"{target} pose"
            target = self.poses[target]
        if label:
            print(f"[arm] {label}")
        for ch in range(5):
            self.move_channel(ch, target[ch])

    def goto_stacked(self, target, label: str = "") -> None:
        """Like goto(), but moves CH5 back to CH1 to fold the arm."""
        if isinstance(target, str):
            if target not in self.poses:
                raise ValueError(f"unknown pose '{target}'; "
                                 f"known: {list(self.poses)}")
            label = label or f"{target} pose"
            target = self.poses[target]
        if label:
            print(f"[arm] {label}")
        for ch in reversed(range(5)):
            self.move_channel(ch, target[ch])

    def open_gripper(self) -> None:
        self.move_channel(5, self.profile.gripper_open)

    def close_gripper(self) -> None:
        self.move_channel(5, self.profile.gripper_closed)

    def jog_channel(self, ch: int, delta_deg: float) -> float:
        """Nudge one channel and return its new angle."""
        if not 0 <= ch <= 5:
            raise ValueError(f"channel {ch} out of range 0-5")
        current = self.gripper if ch == 5 else self.arm[ch]
        target = clamp_channel_angle(ch, current + float(delta_deg))
        self.move_channel(ch, target)
        return target

    def release(self) -> None:
        """Turn off all servos so the arm goes limp."""
        try:
            self.actuator.release()
            print("[arm] RELEASED — no PWM, arm is limp (any move re-engages).")
        except Exception as exc:
            print(f"[arm] release failed ({exc})")

    def get_pose(self) -> dict:
        """Current commanded arm pose and gripper angle."""
        return {"arm": list(self.arm), "gripper": self.gripper}

    def print_pose(self) -> None:
        angles = "  ".join(f"{n.split()[0]}={v:.1f}"
                           for n, v in zip(CH_NAMES, self.arm))
        print(f"[pose] {angles}  grip={self.gripper:.1f}")

    def collect(self, solved: list, tin_pose: str = "upright",
                dump: bool = True) -> bool:
        """Grab the can at the solved pose, then drop it in the bin."""
        if solved is None or len(solved) < 5:
            print("[arm] collect: invalid pose, arm NOT moved.")
            return False
        print(f"[arm] collect ({tin_pose}) at CH1-5 = {solved}")
        self.open_gripper()
        for ch in (1, 2, 3):
            self.move_channel(ch, self.poses["lift"][ch])
        for ch in grab_order(tin_pose):
            self.move_channel(ch, float(solved[ch]))
        self.close_gripper()
        self.move_channel(1, self.poses["lift"][1])
        if not dump:
            print("[arm] grabbed — still holding.")
            return True
        self.dump_to_bin()
        self.goto_stacked("home", "returning home (stacked)")
        return True

    def dump_to_bin(self) -> None:
        """Carry the held can to the bin and release it."""
        self.goto("bin", "carrying to the bin")
        self.open_gripper()
        print("[arm] collected — can dropped in the bin.")
