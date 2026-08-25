"""The one arm driver: pose tracking, gentle moves and the tuned arc grasp.
Used by the autonomous stack (ArmExecutor), the web dashboard, and the manual
calibration tools in tests/.

Poses, gripper angles and pacing come from arm_settings; callers retune them
through set_speed() / set_gripper_angles() instead of declaring their own
numbers. Nothing moves on construction — the true pose is unknown
without joint feedback, so no tool may surprise-move the arm.
"""
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
    """Rule 1 & 2: reads solver output and routes it to the actuator
    interface; never talks to I2C/SPI itself.
    Implements src.hardware.actuators.interfaces.ArmPlannerInterface."""

    def __init__(self, actuator: Optional[ArmActuator] = None,
                 profile: Optional[MotionProfile] = None,
                 release_on_start: bool = False):
        """release_on_start cuts PWM immediately (calibration tools: the
        PCA9685 still holds the previous process's angles, so "no movement"
        is not "at rest")."""
        self.actuator = actuator or ArmActuator()
        self.profile = profile or MotionProfile()
        self.poses = {
            "home": list(HOME_ANGLES),
            "lift": list(ARC_LIFT_ARM),
            "bin":  list(BIN_DROP_ANGLES),
            "grab": list(GRAB_ANGLES),
        }
        # A wrong start turns "stepped" moves into full-speed snaps, so prefer
        # the pose persisted by the previous session over assuming home.
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

    # ── Tunables ─────────────────────────────────────────────────────────

    def set_speed(self, step_deg=None, step_delay=None) -> MotionProfile:
        """Retune move pacing (step_deg degrees per step_delay seconds)."""
        return self.profile.set_speed(step_deg, step_delay)

    def set_gripper_angles(self, open_deg=None, closed_deg=None) -> MotionProfile:
        """Retune the open/close gripper angles (clamped to the safe window)."""
        return self.profile.set_gripper_angles(open_deg, closed_deg)

    # ── Moves ────────────────────────────────────────────────────────────

    def move_channel(self, ch: int, value: float) -> None:
        """Ramp ONE channel (0-4 = CH1-5, 5 = gripper) to value, ending with a
        write-through: tracking can be wrong (no joint feedback), and a
        diff-only path silently drops such commands."""
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
        """Move CH1-5 to a pose, one channel at a time.

        target: a name from self.poses ('home' | 'lift' | 'bin' | 'grab') or
        five explicit CH1..CH5 angles. The gripper is left as it is — open or
        close it explicitly, so carrying a can to a pose never drops it.
        """
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
        """Like goto(), but unwinds CH5->CH1 (LIFO) instead of CH1->CH5 --
        folds the arm back the way it extended, so the wrist (CH4) tucks in
        before the elbow (CH3) sweeps, instead of catching on the bin rim."""
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
        """Nudge one channel by delta degrees; returns the new commanded
        angle. Used by the web dashboard's manual arm control."""
        if not 0 <= ch <= 5:
            raise ValueError(f"channel {ch} out of range 0-5")
        current = self.gripper if ch == 5 else self.arm[ch]
        target = clamp_channel_angle(ch, current + float(delta_deg))
        self.move_channel(ch, target)
        return target

    # ── Power ────────────────────────────────────────────────────────────

    def release(self) -> None:
        """Cut PWM to all 6 channels: servos go limp, nothing holds a pose
        against gravity after the process exits. The arm will droop."""
        try:
            self.actuator.release()
            print("[arm] RELEASED — no PWM, arm is limp (any move re-engages).")
        except Exception as exc:
            print(f"[arm] release failed ({exc})")

    def get_pose(self) -> dict:
        """COMMANDED pose for UIs: {'arm': [CH1..CH5], 'gripper': CH6}. Valid
        as long as every move went through this class (servos have no
        feedback)."""
        return {"arm": list(self.arm), "gripper": self.gripper}

    def print_pose(self) -> None:
        angles = "  ".join(f"{n.split()[0]}={v:.1f}"
                           for n, v in zip(CH_NAMES, self.arm))
        print(f"[pose] {angles}  grip={self.gripper:.1f}")

    # ── Collecting ───────────────────────────────────────────────────────

    def collect(self, solved: list, tin_pose: str = "upright",
                dump: bool = True) -> bool:
        """Grab the can at a solved [CH1..CH5] (from ArcGraspSolver), then
        dump_to_bin() unless dump=False (stops after the grab, still HOLDING
        — for checking the grip by hand during calibration).

        tin_pose: "upright" | "lying" | "axial" picks the approach order,
        whose LAST channel lowers onto the can. The base must already be
        held — callers that own a motor driver brake it around this call.
        """
        if solved is None or len(solved) < 5:
            print("[arm] collect: invalid pose, arm NOT moved.")
            return False
        print(f"[arm] collect ({tin_pose}) at CH1-5 = {solved}")
        self.open_gripper()
        # TESTING: skip pre-lift/post-lift, go straight grab -> bin, see outcome.
        # for ch in (1, 2, 3):                         # CH2/CH3/CH4 -> lift high
        #     self.move_channel(ch, self.poses["lift"][ch])
        for ch in grab_order(tin_pose):
            self.move_channel(ch, float(solved[ch]))
        self.close_gripper()
        # self.move_channel(1, self.poses["lift"][1])  # lift shoulder, holding
        if not dump:
            print("[arm] grabbed — still holding.")
            return True
        self.dump_to_bin()
        self.goto_stacked("home", "returning home (stacked)")
        return True

    def dump_to_bin(self) -> None:
        """Carry the held tin to the onboard bin pose and release it there."""
        self.goto("bin", "carrying to the bin")
        self.open_gripper()
        print("[arm] collected — can dropped in the bin.")
