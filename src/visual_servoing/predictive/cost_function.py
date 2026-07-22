"""
src/visual_servoing/cost_function.py

Scores one predicted (action sequence, lateral-error trace, distance trace)
from trajectory_planner.py. Anchored to distance_error.py's own
STOP_DISTANCE_CM so the planner is regulating toward the SAME physical target
the reactive controller used, not some newly invented setpoint.

Pure function, no state: total cost = sum over the horizon of

    W_LATERAL   * (lateral_error_k / CENTER_TOLERANCE)^2
  + W_DISTANCE  * ((distance_k - STOP_DISTANCE_CM) / STOP_DISTANCE_CM)^2
  + W_STEER_OSC * ((steer_k - steer_{k-1}) / MAX_STEER_ANGLE_DEG)^2
  + W_SPEED_OSC * ((speed_k - speed_{k-1}) / MAX_SPEED)^2
  + W_OVERSHOOT * (max(0, MIN_SAFE_DISTANCE_CM - distance_k) / MIN_SAFE_DISTANCE_CM)^2

Every term is divided by its own characteristic scale BEFORE squaring, so
each one lands around O(1) for a "meaningfully bad" deviation (fully
off-center, a full-width steer swing, ...) regardless of what raw units it's
measured in. This matters a lot here: lateral_error lives in [-0.5, 0.5]
while distance lives in tens-to-hundreds of cm and steer swings are tens of
degrees -- summing raw squared values without normalizing lets whichever
quantity happens to have the largest raw units (distance, at any real range)
silently swamp the others, so the search always "declines to steer" no
matter how off-center the target is, since a normalized-unaware oscillation
term would also make ANY steering look expensive relative to an unnormalized
distance term. Normalizing first is what makes the weights below mean what
they say.

The distance term is centered on STOP_DISTANCE_CM rather than on zero, so
"stay still" is only cost-optimal once the robot is actually centered and at
the grasp distance -- the cost function drives progress by itself, with no
separate reward term needed. steer_{-1}/speed_{-1} for k=0 come from the
PREVIOUS control cycle's executed action, so oscillation is penalized across
cycles too, not just within one plan.
"""
from __future__ import annotations

from typing import List

from src.visual_servoing.distance_error import STOP_DISTANCE_CM, CENTER_TOLERANCE
from ..reactive_controller import MAX_STEER_ANGLE_DEG, MAX_SPEED
from .prediction_model import ActionStep

# TODO tune. All O(1): each is "how many times as important as a
# full-scale swing in that one quantity", now that every term is normalized.
W_LATERAL = 1.0      # lateral centering
W_DISTANCE = 1.0     # closing/holding the grasp distance
W_STEER_OSC = 0.1    # smoothness regularizer, not a primary objective
W_SPEED_OSC = 0.05   # smoothness regularizer, not a primary objective
W_OVERSHOOT = 3.0     # safety margin -- weighted well above the primary terms

MIN_SAFE_DISTANCE_CM = STOP_DISTANCE_CM * 0.6  # TODO tune: overshoot threshold


def evaluate(actions: List[ActionStep], predicted_lateral: List[float],
             predicted_distance: List[float], prev_action: ActionStep) -> float:
    cost = 0.0
    last_steer = prev_action.steer_deg
    last_speed = prev_action.speed

    for action, lateral_error, distance_cm in zip(actions, predicted_lateral, predicted_distance):
        cost += W_LATERAL * (lateral_error / CENTER_TOLERANCE) ** 2
        cost += W_DISTANCE * ((distance_cm - STOP_DISTANCE_CM) / STOP_DISTANCE_CM) ** 2
        cost += W_STEER_OSC * ((action.steer_deg - last_steer) / MAX_STEER_ANGLE_DEG) ** 2
        cost += W_SPEED_OSC * ((action.speed - last_speed) / MAX_SPEED) ** 2

        overshoot = max(0.0, MIN_SAFE_DISTANCE_CM - distance_cm)
        cost += W_OVERSHOOT * (overshoot / MIN_SAFE_DISTANCE_CM) ** 2

        last_steer = action.steer_deg
        last_speed = action.speed

    return cost
