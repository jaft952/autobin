"""Fake wheel driver that prints commands instead of moving."""
from __future__ import annotations


class PrintActuator:
    """Prints wheel commands, no hardware needed."""

    def apply(self, cmd) -> None:
        print(f"   [wheels] L={cmd.left_speed:+6.1f}  R={cmd.right_speed:+6.1f}  ({cmd.trim_set})")

    def stop(self) -> None:
        print("   [wheels] STOP (coast)")

    def brake(self) -> None:
        print("   [wheels] BRAKE (hold)")

    def close(self) -> None:
        pass
