"""
tests/diagnose_imports.py

Pinpoints which native library causes "Illegal instruction" (SIGILL) on the
Raspberry Pi. Each module is imported in a SEPARATE subprocess, so a crash in
one does not stop the others. Run on the Pi:

    python3 tests/diagnose_imports.py

A line ending in "CRASHED (signal SIGILL ...)" is the culprit. Then try the
suggested fixes printed at the end.
"""

import subprocess
import sys
import platform
import signal


# Imported in dependency order so the first crash points at the lowest-level lib.
MODULES = [
    ("numpy", "import numpy; print(numpy.__version__)"),
    ("cv2", "import cv2; print(cv2.__version__)"),
    ("torch", "import torch; print(torch.__version__)"),
    ("torchvision", "import torchvision; print(torchvision.__version__)"),
    ("ultralytics", "from ultralytics import YOLO; import ultralytics; print(ultralytics.__version__)"),
]


def _signal_name(code: int) -> str:
    """code is negative when the child was killed by a signal (POSIX)."""
    try:
        return signal.Signals(-code).name
    except (ValueError, AttributeError):
        return f"signal {-code}"


def check(name: str, code: str):
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        print(f"  OK    {name:<12} v{proc.stdout.strip()}")
    elif proc.returncode < 0:
        sig = _signal_name(proc.returncode)
        marker = " <-- THIS IS THE CULPRIT" if sig == "SIGILL" else ""
        print(f"  CRASH {name:<12} killed by {sig} (rc={proc.returncode}){marker}")
        if proc.stderr.strip():
            print(f"        stderr: {proc.stderr.strip().splitlines()[-1]}")
    else:
        # Normal Python error (e.g. ModuleNotFoundError) — not a SIGILL.
        last = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "?"
        print(f"  FAIL  {name:<12} exit {proc.returncode}: {last}")


def main():
    print("=" * 60)
    print("System info")
    print("-" * 60)
    print(f"  machine     : {platform.machine()}")       # aarch64 = 64-bit, armv7l = 32-bit
    print(f"  platform    : {platform.platform()}")
    print(f"  python      : {platform.python_version()} ({platform.architecture()[0]})")
    print(f"  executable  : {sys.executable}")
    print("=" * 60)
    print("Importing each native library in its own process:")
    print("-" * 60)
    for name, code in MODULES:
        check(name, code)
    print("=" * 60)
    print("If a line above says SIGILL, try (in order):")
    print("  1) numpy/cv2 crash  -> run with:  OPENBLAS_CORETYPE=ARMV8 python3 ...")
    print("                         or persist: echo 'export OPENBLAS_CORETYPE=ARMV8' >> ~/.bashrc")
    print("  2) still crashing   -> reinstall the offending lib for THIS arch:")
    print("                         pip uninstall numpy opencv-python && pip install numpy opencv-python")
    print("  3) torch crash      -> install a Pi-compatible build (see notes from Claude)")
    print("=" * 60)


if __name__ == "__main__":
    main()
