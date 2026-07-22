"""
src/visual_servoing/trajectory_planner.py

Deterministic beam search over multi-step (steer, speed) action sequences.
Deliberately NOT random-shooting MPC: this repo's visual_servoing tests
assert exact behavior from synthetic fixtures (see the docstrings in
distance_error.py and approach_drive.py), so the planner has to be
reproducible for a given TargetError -- same input, same plan, every time.

Every candidate step is converted to a WheelCommand through the SAME
DifferentialKinematics.arc_forward_left/right methods the real hardware is
driven with (action_to_command, below), so whatever the search simulates is
guaranteed to be physically reachable -- there is no separate "planning
model" that could drift from what actually gets executed.

Search: starting from a single empty candidate, expand every surviving
partial sequence by every candidate action, score the resulting complete
partial trajectories with cost_function.evaluate(), and keep only the
BEAM_WIDTH cheapest before expanding again. This still lets the action
change from step to step (unlike holding one action for the whole horizon),
while staying far cheaper than the full combinatorial tree.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.visual_servoing.distance_error import TargetError
from src.visual_servoing.approach_drive import MAX_SPEED, MAX_STEER_ANGLE_DEG
from src.visual_servoing.prediction_model import (
    ActionStep, RobotState, TargetPoint, CONTROL_DT_S,
    estimate_initial_state, step as physics_step, predicted_error,
)
from src.visual_servoing.cost_function import evaluate
from src.motion.differential_kinematics import DifferentialKinematics, WheelCommand

HORIZON_STEPS = 5   # TODO tune: ~1s of lookahead at CONTROL_DT_S=0.2s
BEAM_WIDTH = 6       # TODO tune: partial sequences kept alive between steps

STEER_CANDIDATES_DEG = (-MAX_STEER_ANGLE_DEG, -MAX_STEER_ANGLE_DEG / 2, 0.0,
                         MAX_STEER_ANGLE_DEG / 2, MAX_STEER_ANGLE_DEG)  # TODO tune

# No 0.0 here on purpose: the search only ever runs mid-approach (predictive_
# controller.py short-circuits too_close/reached before planning starts), so
# a literal stop is never actually in scope, and offering it as a candidate
# creates a bad local optimum -- with limited steering authority, a fast
# closing speed can make predicted lateral error look WORSE within a short
# horizon than just before starting to turn, and quadratic per-step costs
# then rate "stand still" as cheapest even though it never converges. A
# speed floor (same "motors stall below this" reasoning as
# layer2_approach.APPROACH_MIN_SPEED) removes that trap and keeps every
# candidate making real progress, letting the horizon actually resolve
# whether a shallower or sharper turn tracks the target better.
SPEED_FRACTIONS = (0.3, 0.55, 0.8, 1.0)  # TODO tune, of MAX_SPEED


@dataclass
class Plan:
    actions: List[ActionStep]
    predicted_lateral: List[float]
    predicted_distance: List[float]
    cost: float


@dataclass
class _Candidate:
    actions: List[ActionStep]
    state: RobotState
    predicted_lateral: List[float]
    predicted_distance: List[float]
    cost: float


def action_to_command(action: ActionStep, kin: DifferentialKinematics) -> WheelCommand:
    """Hardware-verified mapping copied from approach_drive.compute_drive_command:
    +ve steer (correcting toward a can right of center) -> arc_forward_LEFT,
    -ve -> arc_forward_RIGHT. Do not re-derive this from geometry -- it was
    flipped from the naive reading once already on real hardware."""
    if action.steer_deg > 0:
        return kin.arc_forward_left(angle_deg=abs(action.steer_deg), speed=action.speed)
    if action.steer_deg < 0:
        return kin.arc_forward_right(angle_deg=abs(action.steer_deg), speed=action.speed)
    return kin.arc_forward_left(angle_deg=0.0, speed=action.speed)


def _candidate_actions() -> List[ActionStep]:
    return [ActionStep(steer_deg=steer, speed=fraction * MAX_SPEED)
            for steer in STEER_CANDIDATES_DEG
            for fraction in SPEED_FRACTIONS]


def plan(error: TargetError, kin: DifferentialKinematics, prev_action: ActionStep,
         horizon: int = HORIZON_STEPS, beam_width: int = BEAM_WIDTH,
         dt_s: float = CONTROL_DT_S) -> Plan:
    robot0, target = estimate_initial_state(error)
    actions_grid = _candidate_actions()

    beam = [_Candidate(actions=[], state=robot0, predicted_lateral=[],
                        predicted_distance=[], cost=0.0)]

    for _ in range(horizon):
        expanded: List[_Candidate] = []
        for candidate in beam:
            for action in actions_grid:
                cmd = action_to_command(action, kin)
                new_state = physics_step(candidate.state, cmd, dt_s, kin.cal)
                lateral_error, distance_cm = predicted_error(new_state, target)

                new_actions = candidate.actions + [action]
                new_lateral = candidate.predicted_lateral + [lateral_error]
                new_distance = candidate.predicted_distance + [distance_cm]
                cost = evaluate(new_actions, new_lateral, new_distance, prev_action)

                expanded.append(_Candidate(
                    actions=new_actions, state=new_state,
                    predicted_lateral=new_lateral, predicted_distance=new_distance,
                    cost=cost,
                ))
        expanded.sort(key=lambda c: c.cost)
        beam = expanded[:beam_width]

    best = beam[0]
    return Plan(actions=best.actions, predicted_lateral=best.predicted_lateral,
                predicted_distance=best.predicted_distance, cost=best.cost)
