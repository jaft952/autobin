"""ArmExecutor: runs the winning arm_action (stow/deploy/hold/grab_arc) as real arm motion. Grabs block the tick and have a cooldown so a still-visible tin doesn't re-trigger."""
from __future__ import annotations

import logging
import time
from typing import Optional

import src.log_levels  # noqa: F401 -- registers log.success()/log.fail()
from src.hardware.actuators.interfaces import ArmPlannerInterface
from src.subsumption.arbitrator import ActionCommand

log = logging.getLogger("arm_executor")

GRAB_COOLDOWN_S = 3.0   # match tests/test_ibvs_centering.py


class ArmExecutor:
    """Executes the winning command's arm_action. One per robot."""

    def __init__(self, planner: Optional[ArmPlannerInterface] = None,
                 sensors=None, grab_zone_check=None, resume_camera=None) -> None:
        if planner is None:
            # Lazy import: wheels-only setups skip the arm stack.
            from src.arm.grasp_planner import GraspPlanner
            planner = GraspPlanner()
        self.planner: ArmPlannerInterface = planner
        self._sensors = sensors
        self._grab_zone_check = grab_zone_check
        self._resume_camera = resume_camera
        self._at_home = False
        self._force_next_home = True  # boot pose unknown, force home on first command
        self._cooldown_until = 0.0

    # ── Dispatch ──────────────────────────────────────────────────────────

    def execute(self, command: ActionCommand) -> None:
        action = command.arm_action
        if action in (None, 'stop'):
            return
        if action in ('stow', 'retract', 'deploy', 'hold'):
            # 'deploy' maps to home too: home is this arm's only travel pose.
            self._ensure_home()
            return
        if action == 'grab_arc':
            params = command.arm_params or {}
            pose = params.get('pose')
            tin_pose = params.get('tin_pose', 'upright')
            self._grab(lambda: self.planner.collect(pose, tin_pose=tin_pose))
            return
        print(f"[ArmExecutor] unknown arm_action '{action}' ignored")

    # ── Internals ─────────────────────────────────────────────────────────

    def _ensure_home(self) -> None:
        if self._force_next_home:
            # First home since boot: open gripper into a known state.
            self._home()
            self._force_next_home = False
            self._at_home = True
            return
        if not self._at_home:
            # release=False: never drop whatever the gripper may be carrying.
            self._home(release=False)
            self._at_home = True

    def _home(self, release: bool = True) -> None:
        self.planner.goto("home")
        if release:
            self.planner.open_gripper()

    def _grab(self, grab_fn) -> None:
        """Run one collection (grab_fn grabs AND dumps), then home."""
        now = time.monotonic()
        if now < self._cooldown_until:
            return
        if self._force_next_home:
            # Never grab from an unknown boot pose; home first.
            self._ensure_home()
        self._at_home = False
        try:
            grabbed = grab_fn()
            self._home()
            self._at_home = True
            if self._resume_camera is not None:
                # Resume camera and wait for a fresh frame before checking the grab zone.
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
            # Cooldown even on failure to avoid re-triggering every tick.
            self._cooldown_until = time.monotonic() + GRAB_COOLDOWN_S
