"""
src/hardware/sensors/battery_sensor.py

Reads the 3S (three-cell) 18650 pack voltage via an INA219 (I2C
voltage/current sensor chip) wired in series between the battery and the
buck converter / motor driver. Percentage is a linear map over the pack's
usable range — not the true Li-ion discharge curve, but close enough for a
dashboard gauge.

If the INA219 isn't on the bus (not wired yet, or a bench run with no
battery), init raises and the caller falls back to the placeholder — this
class never guesses a fake reading.
"""
from __future__ import annotations

# 3S Li-ion pack: 4.2V/cell full, 3.3V/cell empty (stops short of the
# 3.0V/cell hard cutoff to protect the cells).
CELL_COUNT = 3
FULL_V = 4.2 * CELL_COUNT
EMPTY_V = 3.3 * CELL_COUNT

# INA219 default I2C address is 0x40, same as the PCA9685 — the INA219
# board's A0/A1 solder pads must be bridged to move it off that address.
DEFAULT_ADDRESS = 0x41


class BatterySensor:
    def __init__(self, i2c_bus: int = 1, address: int = DEFAULT_ADDRESS):
        from ina219 import INA219    # pip install pi-ina219 (Pi only)
        self._ina = INA219(shunt_ohms=0.1, busnum=i2c_bus, address=address)
        self._ina.configure()

    def get_voltage(self) -> float:
        return self._ina.voltage()

    def get_current_ma(self) -> float:
        return self._ina.current()

    def get_battery_level(self) -> float:
        """0.0-1.0, linearly mapped over the pack's usable voltage range."""
        v = self.get_voltage()
        pct = (v - EMPTY_V) / (FULL_V - EMPTY_V)
        return max(0.0, min(1.0, pct))
