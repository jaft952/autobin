"""Actuator contracts (mirrors src/hardware/sensors/interfaces.py).

Real drivers, print stubs and test fakes all implement these structurally
(typing.Protocol — no inheritance required), so callers can substitute any
of them without checking type.
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class ActuatorInterface(Protocol):
    """Wheel actuator surface (PWMActuator, PrintActuator, ...)."""

    def apply(self, command) -> None: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...


@runtime_checkable
class ArmPlannerInterface(Protocol):
    """Arm driver surface (GraspPlanner, FakePlanner, PrintPlanner, ...)."""

    def goto(self, target, label: str = "") -> None: ...
    def open_gripper(self) -> None: ...
    def close_gripper(self) -> None: ...
    def collect(self, solved: list, tin_pose: str = "upright",
                dump: bool = True) -> bool: ...
    def dump_to_bin(self) -> None: ...
