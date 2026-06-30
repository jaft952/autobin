"""
src/visual_servoing/chassis_controller.py

Chassis layer — turns IBVS centering suggestions into real wheel motion.

IBVSCentering produces a discrete ChassisMove each tick (FORWARD / TURN_LEFT /
FORWARD_LEFT / HOLD / SEARCH ...). This class is the bridge from that suggestion
to the differential-drive base:

    ChassisMove --DifferentialKinematics--> WheelCommand --PWMActuator--> motors

Keeping it here (not inside ibvs_centering.py) preserves the existing layering:
ibvs_centering stays hardware-free and unit-testable, src/motion stays a pure
hardware-agnostic package, and this orchestration module is the only chassis
code that touches GPIO.

--- MOTION STRATEGY (steering mode) ---
Earlier versions used discrete ChassisMove states (TURN_LEFT = spin in place,
FORWARD = straight) driven either continuously (overshoots through YOLO's slow
inference window) or as short pulses (jerky, stop-start).

This version computes wheel speeds DIRECTLY from the raw (error_x, error_y)
signal, like a car's steering wheel:

    turn_component    = error_x * STEER_GAIN   (how sharp to curve)
    forward_component = base speed scaled by how far the tin is (error_y)

    left_speed  = forward_component - turn_component
    right_speed = forward_component + turn_component

This means the base drives in a continuous ARC toward the tin instead of
spinning in place or pulsing. Large error_x -> sharp curve. Small error_x ->
gentle curve, blending smoothly into straight-line forward motion as the tin
centers. No sleep, no discrete pulse — motors run continuously and the curve
itself tightens as the error shrinks, so it settles instead of overshooting.

BACKWARD / SEARCH / HOLD still use the original discrete kinematics since
"steering" doesn't apply to those (you don't arc in reverse toward a tin
that's too close, and SEARCH/HOLD have no directional component at all).
"""
from __future__ import annotations

import time
from typing import Callable, Dict, Optional

from src.motion.calibration import MotionCalibration, MotorPins
from src.motion.differential_kinematics import DifferentialKinematics, WheelCommand
from src.visual_servoing.ibvs_centering import ChassisMove

# --- Steering mode constants (used by drive_toward_target) ---
# error_x and error_y are normalized roughly -1..1 (0 = perfectly centered).
#
# STEER_BASE_SPEED   -> forward speed fraction (0..1) when error_x is ~0 (driving
#                       straight). This is the "cruising" speed toward the tin.
# STEER_MIN_SPEED    -> floor so the base never crawls so slow it stalls.
# STEER_GAIN         -> how many speed-fraction points of turn per unit of error_x.
#                       Bigger -> sharper curves for the same off-center amount.
# STEER_MAX_TURN     -> caps how much differential is allowed, so a wheel never
#                       reverses direction unexpectedly when fully off-center;
#                       it just slows that side down toward (but not past) zero.
# Tune on the Pi:
#   STEER_GAIN too low    -> base barely curves, drifts past the tin sideways
#   STEER_GAIN too high   -> curves too sharply, may overshoot the other way
#   STEER_BASE_SPEED too high -> approaches too fast, overshoots distance-wise
STEER_BASE_SPEED = 0.55
STEER_MIN_SPEED  = 0.35
STEER_GAIN       = 0.9
STEER_MAX_TURN   = 0.55

# Distance scaling: error_y (far = near top of frame = more negative in this
# codebase's convention -> "FORWARD") scales the base forward speed so a tin
# that's far away gets a brisker approach, and a tin that's almost at the
# right distance gets a gentle creep.
STEER_DIST_GAIN = 1.2

# --- Legacy discrete-move constants (still used for BACKWARD/SEARCH/HOLD and
#     for apply()/pulse() callers that haven't switched to steering mode) ---
SPEED_MIN = 0.70
SPEED_MAX = 1.0
SPEED_MAX_TURN = 0.45
SPEED_GAIN = 3
SPEED_MAX_BACKWARD = 0.7

DRIVE_PULSE_MIN = 0.05
DRIVE_PULSE_MAX = 0.15
DRIVE_PULSE_GAIN = 1.0
DRIVE_PULSE_S = 0.12


class ChassisController:
    """Drives the differential base to satisfy an IBVS ChassisMove suggestion.

    For FORWARD / TURN_LEFT / TURN_RIGHT / FORWARD_LEFT / FORWARD_RIGHT, prefer
    drive_toward_target(error_x, error_y) for smooth continuous arcing motion.
    BACKWARD / SEARCH / HOLD still go through apply_for_error()/apply() as
    discrete moves since steering doesn't apply to them.
    """

    def __init__(
        self,
        actuator=None,
        calibration: Optional[MotionCalibration] = None,
        pins: Optional[MotorPins] = None,
        kinematics: Optional[DifferentialKinematics] = None,
    ) -> None:
        self.cal = calibration or MotionCalibration()
        self.kin = kinematics or DifferentialKinematics(self.cal)
        if actuator is None:
            from src.hardware.actuators.pwm_driver import PWMActuator
            actuator = PWMActuator(pins=pins, calibration=self.cal)
        self.actuator = actuator

        self._move_table: Dict[ChassisMove, Optional[Callable[[], WheelCommand]]] = {
            ChassisMove.FORWARD: self.kin.forward,
            ChassisMove.BACKWARD: self.kin.backward,
            ChassisMove.TURN_LEFT: self.kin.turn_left,
            ChassisMove.TURN_RIGHT: self.kin.turn_right,
            ChassisMove.FORWARD_LEFT: self.kin.arc_forward_left,
            ChassisMove.FORWARD_RIGHT: self.kin.arc_forward_right,
            ChassisMove.BACKWARD_LEFT: self.kin.arc_backward_left,
            ChassisMove.BACKWARD_RIGHT: self.kin.arc_backward_right,
            ChassisMove.SEARCH: None,
            ChassisMove.HOLD: None,
        }
        self._last: Optional[ChassisMove] = None
        self._current_move: Optional[ChassisMove] = None

    @property
    def last_move(self) -> Optional[ChassisMove]:
        return self._last

    # ------------------------------------------------------------------
    # NEW: continuous steering mode
    # ------------------------------------------------------------------

    def drive_toward_target(self, move: ChassisMove, error_x: float, error_y: float) -> None:
        """Drive in a continuous ARC toward the tin, like a car's steering wheel.

        error_x: negative = tin is left of center, positive = right of center.
        error_y: negative = tin is far (near top of frame) -> drive forward;
                 positive = tin is too close (near bottom)  -> handled as BACKWARD.

        For FORWARD / TURN_LEFT / TURN_RIGHT / FORWARD_LEFT / FORWARD_RIGHT this
        computes wheel speeds directly:

            turn      = clamp(error_x * STEER_GAIN, -STEER_MAX_TURN, STEER_MAX_TURN)
            forward   = clamp(STEER_BASE_SPEED + |error_y| * STEER_DIST_GAIN,
                               STEER_MIN_SPEED, 1.0)
            left      = forward - turn
            right     = forward + turn

        Both wheels keep turning forward (never spin in place) and the curve
        tightens automatically as error_x shrinks, so the base smoothly arcs
        onto the tin instead of overshooting past it. BACKWARD / SEARCH / HOLD
        fall back to the discrete apply_for_error() behaviour since arcing
        doesn't apply to them.
        """
        if move in (ChassisMove.BACKWARD, ChassisMove.BACKWARD_LEFT,
                    ChassisMove.BACKWARD_RIGHT, ChassisMove.SEARCH, ChassisMove.HOLD):
            error_mag = max(abs(error_x), abs(error_y))
            self.apply_for_error(move, error_mag)
            return

        # Forward component: brisker the farther away the tin is, floored so
        # it never crawls to a stall, capped at full speed.
        forward = STEER_BASE_SPEED + abs(error_y) * STEER_DIST_GAIN
        forward = max(STEER_MIN_SPEED, min(1.0, forward))

        # Turn component: signed, positive error_x (tin to the right) should
        # curve the base rightward -> slow the right wheel, speed up the left.
        turn = error_x * STEER_GAIN
        turn = max(-STEER_MAX_TURN, min(STEER_MAX_TURN, turn))

        left_fraction  = forward - turn
        right_fraction = forward + turn

        # Never let a wheel go to/below zero or flip direction from the steer
        # alone — clamp to a small positive floor so both wheels always drive
        # forward (this is what makes it an ARC, not a spin-in-place).
        left_fraction  = max(STEER_MIN_SPEED * 0.3, min(1.0, left_fraction))
        right_fraction = max(STEER_MIN_SPEED * 0.3, min(1.0, right_fraction))

        left_speed  = self.cal.forward_speed * left_fraction
        right_speed = self.cal.forward_speed * right_fraction

        cmd = WheelCommand(left_speed, right_speed, True, "forward")
        self.actuator.apply(cmd)
        self._last = move
        self._current_move = move

    # ------------------------------------------------------------------
    # Legacy / discrete-move API (kept for BACKWARD, SEARCH, HOLD, tests)
    # ------------------------------------------------------------------

    def apply(self, move: ChassisMove) -> None:
        """Drive the base continuously for this move at full calibrated speed.
        HOLD/SEARCH stops. Prefer drive_toward_target() for FORWARD-ish moves.
        """
        make_cmd = self._move_table.get(move)
        if make_cmd is None:
            self.stop()
        else:
            self.actuator.apply(make_cmd())
            self._last = move
            self._current_move = move

    def apply_for_error(self, move: ChassisMove, error_mag: float) -> None:
        """Discrete-move proportional speed control (legacy path).

        Kept for BACKWARD / SEARCH / HOLD (steering doesn't apply to these)
        and for any caller not yet using drive_toward_target().
        """
        make_cmd = self._move_table.get(move)
        if make_cmd is None:
            self.stop()
            return

        if move in (ChassisMove.BACKWARD, ChassisMove.BACKWARD_LEFT, ChassisMove.BACKWARD_RIGHT):
            speed_max = SPEED_MAX_BACKWARD
        elif move in (ChassisMove.TURN_LEFT, ChassisMove.TURN_RIGHT):
            speed_max = SPEED_MAX_TURN
        else:
            speed_max = SPEED_MAX

        speed = max(SPEED_MIN, min(speed_max, error_mag * SPEED_GAIN))

        cmd = make_cmd()
        cmd.left_speed *= speed
        cmd.right_speed *= speed

        self.actuator.apply(cmd)
        self._last = move
        self._current_move = move

    def pulse_for_error(self, move: ChassisMove, error_mag: float) -> None:
        """Legacy pulsed method, kept for backward compatibility only."""
        seconds = max(DRIVE_PULSE_MIN, min(DRIVE_PULSE_MAX, error_mag * DRIVE_PULSE_GAIN))
        self.pulse(move, seconds)

    def pulse(self, move: ChassisMove, seconds: float = DRIVE_PULSE_S) -> None:
        """Legacy: move for one short burst then stop. Kept for compatibility."""
        make_cmd = self._move_table.get(move)
        if make_cmd is None:
            self.stop()
            return
        self.actuator.apply(make_cmd())
        self._last = move
        time.sleep(seconds)
        self.stop()

    def stop(self) -> None:
        """Cut motor power and hold position (does not release GPIO)."""
        self.actuator.stop()
        self._last = ChassisMove.HOLD
        self._current_move = None

    def close(self) -> None:
        """Stop and release GPIO. Call once on shutdown."""
        try:
            self.actuator.close()
        finally:
            self._last = None
            self._current_move = None