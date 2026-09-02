"""Zigzag scan state machine: one function per phase transition."""
import enum

from . import tuning
from . import geometry


class Phase(enum.Enum):
    SETTLE     = enum.auto()   # brief halt between phases (direction flips)
    DRIVE      = enum.auto()
    DODGE_TURN = enum.auto()   # pivot away from whatever is ahead
    DODGE_PASS = enum.auto()   # drive past it, watching the side it is on
    DODGE_BACK = enum.auto()   # pivot back onto the lane heading
    TURN1      = enum.auto()
    SHIFT      = enum.auto()
    TURN2      = enum.auto()


def settle(layer, now, front, wall_ahead, sensors):
    if layer._elapsed(now) >= tuning.SETTLE_S:
        layer._enter(layer._next_phase, now)


def drive(layer, now, front, wall_ahead, sensors):
    if wall_ahead:
        layer._start_dodge(sensors, now)
    elif (now - layer._lane_started) >= layer._lane_limit_s:
        layer._turn_left = layer._pivot_side(sensors)
        layer._settle(Phase.TURN1, now)


def dodge_turn(layer, now, front, wall_ahead, sensors):
    if layer._elapsed(now) >= layer.turn_90_s:
        layer._settle(Phase.DODGE_PASS, now)


def dodge_pass(layer, now, front, wall_ahead, sensors):
    elapsed = layer._elapsed(now)
    if elapsed >= tuning.DODGE_MAX_S or wall_ahead:
        # Cannot get round it: treat it as a wall. The dodge already IS the
        # first pivot and the sideways hop, so only TURN2 is left to run.
        layer._turn_left = layer._dodge_left
        layer._settle(Phase.TURN2, now)
    elif elapsed >= tuning.DODGE_MIN_S and geometry.dodge_side_clear(sensors, layer._dodge_left):
        layer._settle(Phase.DODGE_BACK, now)


def dodge_back(layer, now, front, wall_ahead, sensors):
    if layer._elapsed(now) >= layer.turn_90_s:
        layer._settle(Phase.DRIVE, now)


def turn1(layer, now, front, wall_ahead, sensors):
    if layer._elapsed(now) >= layer._turn_s:
        layer._settle(Phase.SHIFT, now)


def shift(layer, now, front, wall_ahead, sensors):
    # Wall ahead during the shift -> skip straight to the second pivot.
    if wall_ahead or layer._elapsed(now) >= layer.shift_s:
        layer._settle(Phase.TURN2, now)


def turn2(layer, now, front, wall_ahead, sensors):
    if layer._elapsed(now) >= layer._turn_s:
        layer._turn_left = not layer._turn_left   # alternate -> zigzag
        layer._start_lane(now)
        layer._settle(Phase.DRIVE, now)


HANDLERS = {
    Phase.SETTLE: settle,
    Phase.DRIVE: drive,
    Phase.DODGE_TURN: dodge_turn,
    Phase.DODGE_PASS: dodge_pass,
    Phase.DODGE_BACK: dodge_back,
    Phase.TURN1: turn1,
    Phase.SHIFT: shift,
    Phase.TURN2: turn2,
}
