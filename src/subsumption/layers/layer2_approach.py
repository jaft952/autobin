"""Layer 2: drive toward the locked can."""
from __future__ import annotations

import time
from typing import Any, Optional

from src.subsumption.layers.base_layer import BaseLayer
from src.subsumption.arbitrator import ActionCommand
from src.arm.arc_grasp import ArcGraspSolver, BAND_TOO_CLOSE
from src.arm.grasp_reach import solve_reach
from src.motion.calibration import MotionCalibration
from src.motion.differential_kinematics import DifferentialKinematics, WheelCommand
import src.visual_servoing.reactive_controller as reactive_mod
from src.visual_servoing.reactive_controller import compute_reactive_command

_CAL = MotionCalibration()
_KIN = DifferentialKinematics(_CAL)

RETREAT_GAP_S = 0.05
RETREAT_PULSE_S = 0.3
LOST_GRACE_S = 1.0


def _to_motion_vector(wheels: WheelCommand):
    """Convert wheel speeds back into a motion vector."""
    scale = 2.0 * _CAL.forward_speed
    v_x = (wheels.left_speed + wheels.right_speed) / scale
    v_theta = (wheels.right_speed - wheels.left_speed) / scale
    return (v_x, 0, v_theta)


class ApproachLitterLayer(BaseLayer):
    """Layer 2: steers the robot toward a detected can."""

    def __init__(self, arc_solver: Optional[ArcGraspSolver] = None):
        super().__init__(layer_id=2)
        self.solver = arc_solver or ArcGraspSolver()
        self._retreating: bool = False
        self._retreat_until: float = 0.0
        self._was_too_close: bool = False
        self._lost_until: float = 0.0
        self._suppressed_since: Optional[float] = None

    def reset(self) -> None:
        self._retreating = False
        self._retreat_until = 0.0
        self._was_too_close = False
        self._lost_until = 0.0
        self._suppressed_since = None

    def notify_arbitration(self, won: bool) -> None:
        if not won and self._suppressed_since is None:
            self._suppressed_since = time.monotonic()

    def evaluate(self, sensors: Any) -> ActionCommand:
        now = time.monotonic()
        self._absorb_suppressed_time(now)

        error = sensors.get_litter_target_error()
        if error is None or not error.found:
            if self._was_too_close and now < self._lost_until:
                return self._stop("OVERSHOT - waiting for the detection")
            self._was_too_close = False
            return ActionCommand(layer_id=self.layer_id, active=False)
        self._lost_until = now + LOST_GRACE_S

        reach = solve_reach(self.solver, sensors)

        if reach.band == BAND_TOO_CLOSE:
            return self._retreat_pulse(now)
        self._was_too_close = False

        if reach.pose is not None:
            nx, ny = reach.point  # type: ignore
            return self._stop(f"IN REACH, holding ({reach.klass} "
                              f"nx={nx:.2f} ny={ny:.2f})")

        wheels = compute_reactive_command(error, _KIN)
        if wheels is None:
            return self._stop(f"REACHED but not in reach "
                              f"(dist={error.distance_cm}, band={reach.band})")

        label = "TOO CLOSE, backing off" if error.too_close else \
            f"APPROACHING LITTER (ex={error.lateral_error:+.2f}, dist={error.distance_cm})"
        return ActionCommand(
            layer_id=self.layer_id,
            active=True,
            motion_vector=_to_motion_vector(wheels),
            arm_action='deploy',
            message=label,
        )

    def _absorb_suppressed_time(self, now: float) -> None:
        """Do not count time paused by a higher layer."""
        if self._suppressed_since is None:
            return
        paused = now - self._suppressed_since
        if self._retreat_until:
            self._retreat_until += paused
        if self._lost_until:
            self._lost_until += paused
        self._suppressed_since = None

    def _retreat_pulse(self, now: float) -> ActionCommand:
        """Back up a short distance."""
        if not self._was_too_close:
            self._was_too_close = True
            self._retreating = False
            self._retreat_until = now + RETREAT_GAP_S
        elif now >= self._retreat_until:
            self._retreating = not self._retreating
            self._retreat_until = now + (RETREAT_PULSE_S if self._retreating
                                         else RETREAT_GAP_S)

        if self._retreating:
            v_x = -reactive_mod.BACKUP_SPEED / _CAL.forward_speed
            return ActionCommand(
                layer_id=self.layer_id, active=True,
                motion_vector=(v_x, 0, 0), arm_action='deploy',
                message="OVERSHOT past nearest arc - retreat pulse",
            )
        return self._stop("OVERSHOT past nearest arc - pulse gap")

    def _stop(self, message: str) -> ActionCommand:
        """Stop the robot and park the arm."""
        return ActionCommand(
            layer_id=self.layer_id,
            active=True,
            motion_vector=(0, 0, 0),
            arm_action='deploy',
            message=message,
        )
