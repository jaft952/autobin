"""
src/subsumption/arm_executor.py

ArmExecutor — turns the Arbitrator's winning ActionCommand.arm_action into
real arm motion, the arm-side twin of MotionExecutor.

Layers only emit semantic strings + payload (Rule 1/2); after arbitration the
main loop hands the winning command here:

    arm_action            arm_params                what happens
    ----------            ----------                ------------
    'stow'/'retract'/     -                         ensure the arm is at the
    'deploy'                                        home/travel pose (no-op if
                                                    already there)
    'grab_arc'            {'pose': [CH1..CH5]}      tuned arc sequence at the
                                                    solved pose, then dump into
                                                    the onboard bin, then home
    'grab_ik'             {'target_m': (x, y)}      IK move_to on the floor
                                                    point, close, dump, home
    'stop' / None         -                         nothing (arm moves are
                                                    blocking; can't interrupt)

BLOCKING: a grab is a multi-second servo sequence and runs inline, so the
subsumption tick pauses during it. That is acceptable because every grab
command also carries motion_vector (0,0,0) — the base is already halted —
and sensor readings would be of a scene the arm is occluding anyway.

A cooldown after each grab keeps a still-visible tin (mid-lift, or a failed
grab) from re-triggering the sequence on the very next tick.

GraspPlanner is imported lazily: its import chain pulls in ikpy, which dev
machines may not have. Pass a fake planner for logic tests.
"""
from __future__ import annotations

import time
from typing import Optional

from src.subsumption.arbitrator import ActionCommand

GRAB_COOLDOWN_S = 4.0

# IK-fallback grab height: aim the gripper this far above the floor so the
# jaws wrap the tin's body instead of scraping the ground. Tune on the Pi.
GRAB_HEIGHT_ABOVE_FLOOR_M = 0.03


class ArmExecutor:
    """Executes the winning command's arm_action. One per robot."""

    def __init__(self, planner=None) -> None:
        if planner is None:
            # Deferred so wheels-only setups never touch the ikpy import chain.
            from src.arm.grasp_planner import GraspPlanner
            planner = GraspPlanner()
        self.planner = planner
        self._at_home = False
        self._force_next_home = True  # first home since boot must be FORCED:
        #   the arm's true pose is unknown (no feedback) and nothing may move
        #   at server boot — the user starts the system from the dashboard,
        #   and only THEN (first arm command after START) do we assert home.
        self._cooldown_until = 0.0

    # ── Dispatch ──────────────────────────────────────────────────────────

    def execute(self, command: ActionCommand) -> None:
        action = command.arm_action
        if action in (None, 'stop'):
            return
        if action in ('stow', 'retract', 'deploy'):
            # 'deploy' also maps to home: home IS the travel/ready pose on
            # this arm; there is no separate deployed idle posture.
            self._ensure_home()
            return
        if action == 'grab_arc':
            pose = (command.arm_params or {}).get('pose')
            self._grab(lambda: self.planner.grab_arc_pose(pose))
            return
        if action == 'grab_ik':
            target = (command.arm_params or {}).get('target_m')
            if target is not None:
                self._grab(lambda: self._ik_grab(*target))
            return
        print(f"[ArmExecutor] unknown arm_action '{action}' ignored")

    # ── Internals ─────────────────────────────────────────────────────────

    def _ensure_home(self) -> None:
        if self._force_next_home:
            # First home since boot: FORCE-command every servo (we can't
            # trust the tracked pose before the arm has ever been homed).
            # Planners without force_home (test fakes) fall back to home().
            force = getattr(self.planner, "force_home", None)
            (force or self.planner.home)()
            self._force_next_home = False
            self._at_home = True
            return
        if not self._at_home:
            self.planner.home()
            self._at_home = True

    def _grab(self, grab_fn) -> None:
        """Run one full collection: grab -> dump into bin -> home."""
        now = time.monotonic()
        if now < self._cooldown_until:
            return
        if self._force_next_home:
            # Never start a grab from an unknown boot pose — assert home
            # first (covers "tin already grabbable on the very first tick").
            self._ensure_home()
        self._at_home = False
        try:
            if grab_fn():
                self.planner.dump_to_bin()
            self.planner.home()
            self._at_home = True
        finally:
            # Cooldown even on failure so an unreachable/missed tin doesn't
            # re-trigger the whole sequence every tick.
            self._cooldown_until = time.monotonic() + GRAB_COOLDOWN_S

    def _ik_grab(self, x_m: float, y_m: float) -> bool:
        """Model-IK fallback grab at a floor point (arm frame, meters)."""
        from src.arm.grasp_planner import DECK_ABOVE_FLOOR_M
        z = -DECK_ABOVE_FLOOR_M + GRAB_HEIGHT_ABOVE_FLOOR_M
        self.planner.control_gripper("open")
        if not self.planner.move_to([x_m, y_m, z]):
            return False                     # unreachable — reported by planner
        self.planner.control_gripper("close")
        return True
