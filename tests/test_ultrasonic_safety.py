import pytest

from src.visual_servoing.ultrasonic_safety import (
    UltrasonicSafety, EMERGENCY_STOP_CM, GRAB_CONFIRM_CM)


class FakeUltrasonicSensor:
    def __init__(self, distance_cm):
        self._distance_cm = distance_cm

    def update(self):
        return None

    def get_distance_cm(self):
        return self._distance_cm

    def close(self):
        return None


def _safety(distance_cm):
    safety = UltrasonicSafety.__new__(UltrasonicSafety)
    safety.sensor = FakeUltrasonicSensor(distance_cm)  # type: ignore
    return safety


def test_can_grab_true_when_within_grab_range_even_if_emergency_stop_active():
    """emergency_stop is a drive-only cutoff; the tin can is EXPECTED to be
    this close at the correct grab position, so it must not block the grab
    (see EMERGENCY_STOP_CM's docstring in ultrasonic_safety.py)."""
    close_enough = min(EMERGENCY_STOP_CM, GRAB_CONFIRM_CM) - 1.0
    safety = _safety(close_enough)

    state = safety.update()

    assert state.grab_confirmed is True
    assert safety.can_grab() is True   # true regardless of emergency_stop's value


def test_can_grab_false_when_too_far():
    too_far = GRAB_CONFIRM_CM + 10.0
    safety = _safety(too_far)

    state = safety.update()

    assert state.grab_confirmed is False
    assert safety.can_grab() is False


def test_can_grab_matches_grab_confirmed_independent_of_emergency_stop():
    """can_grab() must track grab_confirmed alone -- emergency_stop is a
    separate, drive-only signal (see EMERGENCY_STOP_CM's docstring)."""
    for distance in (2.0, min(EMERGENCY_STOP_CM, GRAB_CONFIRM_CM) - 1.0,
                      GRAB_CONFIRM_CM + 10.0):
        safety = _safety(distance)
        state = safety.update()
        assert safety.can_grab() == state.grab_confirmed, (distance, state)
