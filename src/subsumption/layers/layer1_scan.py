"""
Thin entry point kept for debugging/import stability. The real zigzag
scan logic lives in src/scanning/ (tuning.py, geometry.py, phases.py,
layer.py) -- this file just re-exports the layer class.

Tunable constants are NOT re-exported here: tests/runtime monkeypatch
them by module attribute (e.g. `tuning.TIMING_JITTER = 0.0`), which only
works against the module object that phases.py/geometry.py actually read
from. Import them from src.scanning.tuning directly, not from here.
"""
from src.scanning.layer import ScanAroundLayer

__all__ = ["ScanAroundLayer"]
