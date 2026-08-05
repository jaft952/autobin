import pytest

from src.visual_servoing.ultrasonic_safety import UltrasonicSafety


class FakeUltrasonicSensor:
    def __init__(self, distance_cm):
        self._distance_cm = distance_cm

    def update(self):
        return None

    def get_distance_cm(self):
        return self._distance_cm

    def close(self):
        return None


def test_can_grab_is_blocked_when_emergency_stop_is_active():
    safety = UltrasonicSafety.__new__(UltrasonicSafety)
    safety.sensor = FakeUltrasonicSensor(10.0) # type: ignore

    state = safety.update()

    assert state.emergency_stop is True
    assert state.grab_confirmed is True
    assert safety.can_grab() is False


def test_can_grab_when_within_grab_range_but_outside_emergency_stop():
    safety = UltrasonicSafety.__new__(UltrasonicSafety)
    safety.sensor = FakeUltrasonicSensor(20.0) # type: ignore

    state = safety.update()

    assert state.emergency_stop is False
    assert state.grab_confirmed is True
    assert safety.can_grab() is True
