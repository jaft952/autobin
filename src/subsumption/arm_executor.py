"""Runs the winning arm command on the real arm."""
from __future__ import annotations

import logging
import time
from typing import Optional

import src.log_levels  # noqa: F401
from src.hardware.actuators.interfaces import ArmPlannerInterface
from src.subsumption.arbitrator import ActionCommand

log = logging.getLogger("arm_executor")

GRAB_COOLDOWN_S = 3.0


class ArmExecutor:
    """Carries out the arm part of the winning command."""

    def __init__(self, planner: Optional[ArmPlannerInterface] = None,
                 sensors=None, grab_zone_check=None, resume_camera=None) -> None:
        if planner is None:
            from src.arm.grasp_planner import GraspPlanner
            planner = GraspPlanner()
        self.planner: ArmPlannerInterface = planner
        self._sensors = sensors
        self._grab_zone_check = grab_zone_check
        self._resume_camera = resume_camera
        self._at_home = False
        self._force_next_home = True
        self._cooldown_until = 0.0

    def execute(self, command: ActionCommand) -> None:
        action = command.arm_action
        if action in (None, 'stop'):
            return
        if action in ('stow', 'retract', 'deploy', 'hold'):
            self._ensure_home()
            return
        if action == 'grab_arc':
            params = command.arm_params or {}
            pose = params.get('pose')
            tin_pose = params.get('tin_pose', 'upright')
            self._grab(lambda: self.planner.collect(pose, tin_pose=tin_pose))
            return
        print(f"[ArmExecutor] unknown arm_action '{action}' ignored")

    def _ensure_home(self) -> None:
        if self._force_next_home:
            self._home()
            self._force_next_home = False
            self._at_home = True
            return
        if not self._at_home:
            self._home(release=False)
            self._at_home = True

    def _home(self, release: bool = True) -> None:
        self.planner.goto("home")
        if release:
            self.planner.open_gripper()

    def _grab(self, grab_fn) -> None:
        """Run one pickup, then move the arm home."""
        now = time.monotonic()
        if now < self._cooldown_until:
            return
        if self._force_next_home:
            self._ensure_home()
        self._at_home = False
        try:
            grabbed = grab_fn()
            self._home()
            self._at_home = True
            if self._resume_camera is not None:
                self._resume_camera()
                if self._sensors is not None:
                    self._sensors.wait_for_fresh_frames(n=2, timeout=2.0)
            if not grabbed:
                log.fail("grab refused (unreachable/invalid pose)")
            elif self._sensors is not None and self._grab_zone_check is not None:
                if self._grab_zone_check(self._sensors):
                    log.fail("grab sequence completed but a can is still "
                             "in the grab zone — likely missed/knocked aside")
                else:
                    log.success("grab zone clear after collect (unconfirmed "
                                "whether it landed in the bin)")
            else:
                log.success("grab sequence completed (unconfirmed)")
        finally:
            self._cooldown_until = time.monotonic() + GRAB_COOLDOWN_S
