from __future__ import annotations

import time
from typing import Any, Optional

from src.subsumption.layers.base_layer import BaseLayer
from src.subsumption.arbitrator import ActionCommand
from src.arm.arc_grasp import ArcGraspSolver
from src.arm.grasp_reach import solve_reach

GRAB_STABLE_S = 2.0
GRABBABLE_LATCH_S = 2.0


class CollectLitterLayer(BaseLayer):
    """Layer 3: picks up the can with the arm."""

    def __init__(self, arc_solver: Optional[ArcGraspSolver] = None):
        super().__init__(layer_id=3)
        self.solver = arc_solver or ArcGraspSolver()
        self._ready_since: Optional[float] = None
        self._latched_until: float = 0.0
        self._suppressed_since: Optional[float] = None
        self._last_won: bool = False

    def reset(self) -> None:
        self._ready_since = None
        self._latched_until = 0.0
        self._suppressed_since = None
        self._last_won = False

    def notify_arbitration(self, won: bool) -> None:
        self._last_won = won
        if not won and self._suppressed_since is None:
            self._suppressed_since = time.monotonic()

    def evaluate(self, sensors: Any) -> ActionCommand:
        now = time.monotonic()
        self._absorb_suppressed_time(now)
        solved, klass, point, band = self._solve(sensors)

        if solved is None:
            self._ready_since = None
            if band is None and now < self._latched_until:
                return self._stop("GRAB HOLD (detection blinked)")
            return ActionCommand(layer_id=self.layer_id, active=False)

        self._latched_until = now + GRABBABLE_LATCH_S

        if self._ready_since is None:
            self._ready_since = now
        stable_s = now - self._ready_since
        if stable_s < GRAB_STABLE_S:
            return self._stop(f"GRAB HOLD (stabilizing {stable_s:.1f}s"
                              f"/{GRAB_STABLE_S:.0f}s)")

        nx, ny = point  # type: ignore
        if klass == "upright":
            label = "upright"
        elif klass == "axial":
            label = "lying end-on"
        else:
            label = f"lying CH5={solved[4]:.0f}"
        return ActionCommand(
            layer_id=self.layer_id,
            active=True,
            motion_vector=(0, 0, 0),
            arm_action='grab_arc',
            arm_params={'pose': solved, 'tin_pose': klass},
            message=f"ARC GRAB ({label}) @ nx={nx:.2f} ny={ny:.2f}",
        )

    def _absorb_suppressed_time(self, now: float) -> None:
        """Do not count time paused by Layer 4."""
        if self._suppressed_since is None:
            return
        paused = now - self._suppressed_since
        if self._ready_since is not None:
            self._ready_since += paused
        if self._latched_until:
            self._latched_until += paused
        self._suppressed_since = None

    def is_grabbable(self, sensors: Any) -> bool:
        """True if the arm can reach the can now."""
        return self._solve(sensors)[0] is not None

    def is_holding_for_grab(self, sensors: Any) -> bool:
        """True if the can is reachable and the robot is already stopped."""
        return self._last_won and self.is_grabbable(sensors)

    def can_still_in_grab_zone(self, sensors: Any) -> bool:
        """Check if the can is still there after a grab."""
        return self.is_grabbable(sensors)

    def _stop(self, message: str) -> ActionCommand:
        """Stop the robot and park the arm."""
        return ActionCommand(
            layer_id=self.layer_id,
            active=True,
            motion_vector=(0, 0, 0),
            arm_action='hold',
            message=message,
        )

    def _solve(self, sensors: Any):
        """Solve the grasp for the locked can."""
        return solve_reach(self.solver, sensors)
