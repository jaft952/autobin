"""Dry-run stand-in for PWMActuator: prints what WOULD be sent to the wheels.

Used by the --no-motors mode of the scan/subsumption test scripts so the
decision logic can be watched with the motors electrically idle.
"""
from __future__ import annotations


class PrintActuator:
    """Same call surface as PWMActuator, minus the hardware.
    Implements src.hardware.actuators.interfaces.ActuatorInterface."""

    def apply(self, cmd) -> None:
        print(f"   [wheels] L={cmd.left_speed:+6.1f}  R={cmd.right_speed:+6.1f}  ({cmd.trim_set})")

    def stop(self) -> None:
        print("   [wheels] STOP (coast)")

    def brake(self) -> None:
        print("   [wheels] BRAKE (hold)")

    def close(self) -> None:
        pass
