"""
src/subsumption/layers/layer2_approach.py

Layer 2: Approach Litter — steers the base toward detected ground litter
with the same continuous-arc philosophy as ChassisController's steering
mode: turn component proportional to horizontal error, forward speed scaled
by distance, so the base curves onto the tin instead of pivot-then-drive.

Coordinates: get_litter_position() is normalized (x, y) with (0.5, 0.5) at
frame center. x > 0.5 = tin right of center -> steer right = NEGATIVE
v_theta (positive v_theta is CCW/left, see MotionExecutor). Small y = tin
near frame top = far away -> drive brisker; large y = close -> creep, so
Layer 3 gets a stable image to solve the grasp on.

This layer stays active the whole time litter is visible; when the tin
enters a grabbable region Layer 3 activates and subsumes it automatically.
"""
from typing import Any

from src.subsumption.layers.base_layer import BaseLayer
from src.subsumption.arbitrator import ActionCommand
from src.visual_servoing.reactive_controller import FAR_DISTANCE_CM, LOW_DISTANCE_CM

# Tune on the Pi alongside the chassis steering constants:
#   STEER_GAIN too low  -> drifts past the tin sideways
#   STEER_GAIN too high -> S-curves around the centerline
APPROACH_STEER_GAIN = 0.9
APPROACH_MAX_TURN   = 0.5

APPROACH_BASE_SPEED = 0.45   # forward fraction when the tin is mid-frame
APPROACH_DIST_GAIN  = 0.5    # extra speed per unit of "farness" (0.5 - y)
APPROACH_MIN_SPEED  = 0.3    # never crawl below this (motors stall)
APPROACH_MAX_SPEED  = 0.7

# Speed tiers keyed off the monocular distance estimate (same breakpoints as
# reactive_controller, so the two stay in sync as they get re-tuned).
APPROACH_CRUISE_SPEED  = APPROACH_MAX_SPEED
APPROACH_STEP_SPEED    = APPROACH_MIN_SPEED
APPROACH_BACKUP_SPEED  = -0.3


class ApproachLitterLayer(BaseLayer):
    """
    Layer 2: Approach Litter
    Priority: 2 (Low)
    Behavior: Arcs the base toward detected ground litter, slowing as it
              nears, until Layer 3 finds the tin grabbable and takes over.
    """
    def __init__(self):
        super().__init__(layer_id=2)

    def evaluate(self, sensors: Any) -> ActionCommand:
        litter_pos = sensors.get_litter_position()
        if not litter_pos:
            return ActionCommand(layer_id=self.layer_id, active=False)

        x, y = litter_pos
        error_x = x - 0.5                    # +ve = tin right of center

        if sensors.get_litter_too_close():
            return ActionCommand(
                layer_id=self.layer_id,
                active=True,
                motion_vector=(APPROACH_BACKUP_SPEED, 0, 0),
                arm_action='deploy',
                message="TOO CLOSE, backing off",
            )

        steer = -error_x * APPROACH_STEER_GAIN
        steer = max(-APPROACH_MAX_TURN, min(APPROACH_MAX_TURN, steer))

        distance_cm = sensors.get_litter_distance_cm()
        if distance_cm is None:
            # No distance estimate (uncalibrated/degenerate bbox) — fall
            # back to the old y-position proxy for "farness".
            forward = APPROACH_BASE_SPEED + (0.5 - y) * APPROACH_DIST_GAIN
        elif distance_cm > FAR_DISTANCE_CM:
            forward = APPROACH_CRUISE_SPEED
        elif distance_cm > LOW_DISTANCE_CM:
            forward = APPROACH_BASE_SPEED
        else:
            forward = APPROACH_STEP_SPEED
        forward = max(APPROACH_MIN_SPEED, min(APPROACH_MAX_SPEED, forward))

        return ActionCommand(
            layer_id=self.layer_id,
            active=True,
            motion_vector=(forward, 0, steer),
            arm_action='deploy',             # travel pose, ready to grab
            message=f"APPROACHING LITTER (ex={error_x:+.2f}, dist={distance_cm})",
        )
