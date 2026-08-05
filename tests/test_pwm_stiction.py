"""Unit tests for the PWM stiction floor.

MIN_MOVE_DUTY ships at 0.0 (off) and is meant to be raised on hardware, so
these pin the behaviour it will have once someone does — without them the
constant could be bumped on a Pi with nothing checking the scaling rule.
"""
import pytest

from src.hardware.actuators import pwm_driver
from src.hardware.actuators.pwm_driver import _apply_stiction_floor


@pytest.fixture
def floor(monkeypatch):
    monkeypatch.setattr(pwm_driver, "MIN_MOVE_DUTY", 22.0)
    return 22.0


def _ratio(left, right):
    peak = max(abs(left), abs(right))
    return min(abs(left), abs(right)) / peak if peak else 0.0


def test_disabled_by_default_is_a_no_op():
    assert pwm_driver.MIN_MOVE_DUTY == 0.0
    assert _apply_stiction_floor(17.0, 6.9) == (17.0, 6.9)


def test_weak_pair_is_scaled_up_to_the_floor(floor):
    left, right = _apply_stiction_floor(17.0, 6.9)
    assert max(abs(left), abs(right)) == pytest.approx(floor)


def test_scaling_preserves_the_left_right_ratio(floor):
    """The steering signal IS the ratio -- an earlier per-wheel MIN_SPEED
    boost flattened it and made the robot spin instead of arc."""
    before = _ratio(17.0, 6.9)
    left, right = _apply_stiction_floor(17.0, 6.9)
    assert _ratio(left, right) == pytest.approx(before)


def test_inner_wheel_may_stay_below_the_floor(floor):
    """Wanted: a dragging inner wheel is what a tight low-speed turn is."""
    _left, right = _apply_stiction_floor(17.0, 6.9)
    assert abs(right) < floor


def test_a_stopped_pair_stays_stopped(floor):
    assert _apply_stiction_floor(0.0, 0.0) == (0.0, 0.0)


def test_pair_already_above_the_floor_is_untouched(floor):
    assert _apply_stiction_floor(60.0, 24.0) == (60.0, 24.0)


@pytest.mark.parametrize("left,right", [(-15.0, -15.0), (5.0, -5.0), (-5.0, 5.0)])
def test_signs_are_preserved(floor, left, right):
    """Reverse and spin-in-place commands must not be flipped by the scale."""
    new_left, new_right = _apply_stiction_floor(left, right)
    assert (new_left > 0) == (left > 0)
    assert (new_right > 0) == (right > 0)
    assert max(abs(new_left), abs(new_right)) == pytest.approx(floor)
