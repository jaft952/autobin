from __future__ import annotations

import time
from typing import Any, Optional

from src.subsumption.layers.base_layer import BaseLayer
from src.subsumption.arbitrator import ActionCommand
from src.arm.arc_grasp import ArcGraspSolver
from src.arm.grasp_reach import solve_reach

GRAB_CONFIRM_CM = 35.0
GRAB_STABLE_S = 2.0
GRABBABLE_LATCH_S = 2.0


class CollectLitterLayer(BaseLayer):
    """
    Layer 3: Collect Litter
    Priority: 3 (Medium)
    Behavior: Arm only. Once the tin is inside the calibrated arc grid and
              the reading has settled, commands the arc grab. Every command
              it emits is a zero motion vector -- getting the base to a spot
              the arm can reach, and backing out of an overshoot, is Layer
              2's job (src/subsumption/layers/layer2_approach.py).
    """

    def __init__(self, arc_solver: Optional[ArcGraspSolver] = None):
        super().__init__(layer_id=3)
        self.solver = arc_solver or ArcGraspSolver()
        self._ready_since: Optional[float] = None
        self._latched_until: float = 0.0
        self._suppressed_since: Optional[float] = None

    def reset(self) -> None:
        self._ready_since = None
        self._latched_until = 0.0
        self._suppressed_since = None

    # ── Arbitration ───────────────────────────────────────────────────────

    def notify_arbitration(self, won: bool) -> None:
        if not won and self._suppressed_since is None:
            self._suppressed_since = time.monotonic()

    def evaluate(self, sensors: Any) -> ActionCommand:
        now = time.monotonic()
        self._absorb_suppressed_time(now)
        solved, klass, point, band = self._solve(sensors)

        if solved is None:
            self._ready_since = None
            if band is None and now < self._latched_until:
                # No point at all -- a blink, not a departure. Hold the base
                # rather than handing the tin back to Layer 2 to be
                # re-approached from scratch.
                return self._hold("GRAB HOLD (detection blinked)")
            # A band means the tin WAS located, just not reachable from here
            # (too close, too far, off the sampled span). Stand down at once:
            # Layer 2 has to move the base, and every tick spent latched here
            # is a tick it cannot.
            return ActionCommand(layer_id=self.layer_id, active=False)

        self._latched_until = now + GRABBABLE_LATCH_S

        if not self._ultrasonic_confirms(sensors):
            # Hold, don't stand down: a single noisy HC-SR04 ping reading a
            # hair past GRAB_CONFIRM_CM must not hand the wheels back to
            # Layer 2 for even one tick -- its own steering correction is a
            # visible flick, and it moves the base off the spot the arc grid
            # already solved for, right before the grab fires on the next
            # good reading.
            self._ready_since = None
            return self._hold("GRAB HOLD (ultrasonic not yet confirming range)")

        if self._ready_since is None:
            self._ready_since = now
        stable_s = now - self._ready_since
        if stable_s < GRAB_STABLE_S:
            return self._hold(f"GRAB HOLD (stabilizing {stable_s:.1f}s"
                              f"/{GRAB_STABLE_S:.0f}s)")

        nx, ny = point # type: ignore
        if klass == "upright":
            label = "upright"
        elif klass == "axial":
            label = "lying end-on"
        else:
            label = f"lying CH5={solved[4]:.0f}"
        return ActionCommand(
            layer_id=self.layer_id,
            active=True,
            motion_vector=(0, 0, 0),          # halt base for the grab
            arm_action='grab_arc',
            # tin_pose picks the approach order (lying: elbow last)
            arm_params={'pose': solved, 'tin_pose': klass},
            message=f"ARC GRAB ({label}) @ nx={nx:.2f} ny={ny:.2f}",
        )

    def _absorb_suppressed_time(self, now: float) -> None:
        """Open-loop timers must not count time spent suppressed by Layer 5."""
        if self._suppressed_since is None:
            return
        paused = now - self._suppressed_since
        if self._ready_since is not None:
            self._ready_since += paused
        if self._latched_until:
            self._latched_until += paused
        self._suppressed_since = None

    def is_grabbable(self, sensors: Any) -> bool:
        """Can the arc grid solve this tin from where the robot stands?
        Pure — no timers touched, so Layer 5 and ArmExecutor can ask it
        without disturbing this layer's stability clock."""
        return self._solve(sensors)[0] is not None

    def can_still_in_grab_zone(self, sensors: Any) -> bool:
        """Re-run the grabbability check after a grab, once the arm is clear
        of the camera — catches a miss regardless of where a knocked-but-not-
        grabbed can ended up."""
        return self.is_grabbable(sensors)

    # ── Internals ─────────────────────────────────────────────────────────

    def _hold(self, message: str) -> ActionCommand:
        """Base halted, arm parked at the travel pose. The wheels coast --
        there is no brake, see MotionExecutor's HALTING note -- so this
        relies on the gearboxes to keep the base on the spot."""
        return ActionCommand(
            layer_id=self.layer_id,
            active=True,
            motion_vector=(0, 0, 0),
            arm_action='hold',
            message=message,
        )

    def _solve(self, sensors: Any):
        """(solved_or_None, klass, (nx, ny) or None, band). Same solve Layer
        2 arrives on -- one criterion, so the two layers cannot disagree
        about whether the base is in position."""
        return solve_reach(self.solver, sensors)

    @staticmethod
    def _ultrasonic_confirms(sensors: Any) -> bool:
        """Front sensor agrees the tin is within arm range. No reading is NOT
        a veto: an off-center tin (a normal calibrated position) sits outside
        the sensor's narrow beam entirely, and vision plus the arc solver have
        already placed it. A real too-far reading still blocks."""
        getter = getattr(sensors, "get_obstacle_distance_cm", None)
        distance = getter() if getter is not None else None
        return distance is None or distance <= GRAB_CONFIRM_CM
