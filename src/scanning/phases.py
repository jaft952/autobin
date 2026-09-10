"""Zigzag scan state machine: one function per phase transition."""
import enum

from . import tuning
from . import geometry


class Phase(enum.Enum):
    SETTLE = enum.auto()   # brief halt between phases (direction flips)
    DRIVE  = enum.auto()
    TURN1  = enum.auto()
    SHIFT  = enum.auto()
    TURN2  = enum.auto()


def settle(layer, now, front, wall_ahead, sensors):
    if layer._elapsed(now) >= tuning.SETTLE_S:
        layer._enter(layer._next_phase, now)


def drive(layer, now, front, wall_ahead, sensors):
    if wall_ahead or (now - layer._lane_started) >= layer._lane_limit_s:
        layer._turn_left = layer._pivot_side(sensors)
        layer._settle(Phase.TURN1, now)


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
    Phase.TURN1: turn1,
    Phase.SHIFT: shift,
    Phase.TURN2: turn2,
}
