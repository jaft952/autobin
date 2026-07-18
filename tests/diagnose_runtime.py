"""
tests/diagnose_runtime.py

diagnose_imports.py proved every library IMPORTS fine. Yet test_ibvs_centering
and test_arc_grasp still die with "Illegal instruction" (SIGILL) — and crucially
they die AFTER "Model loaded" / "device auto-selected: cpu", i.e. during the
FIRST real computation (the YOLO warmup .predict in detector._load_model), not
at import. So an import-only check can never catch it.

This tool runs escalating RUNTIME operations — numpy math, torch math, a torch
conv (the op YOLO leans on), then a real YOLO .predict — each in its OWN
subprocess. A SIGILL kills the process outright: you cannot try/except it, you
can only launch a child and read the signal it died from. Every stage is run
TWICE: once normally, once with OPENBLAS_CORETYPE=ARMV8 set. So one run tells you
BOTH which operation crashes AND whether that single env var fixes it.

    python3 tests/diagnose_runtime.py
    python3 tests/diagnose_runtime.py --model src/models/best.pt

Reading the grid (OK / SIGILL / FAIL for each stage, plain vs +ARMV8):
  * numpy_matmul is the lowest to crash
        -> OpenBLAS auto-picked the wrong CPU core. If the +ARMV8 column is OK,
           you're done: put  export OPENBLAS_CORETYPE=ARMV8  in ~/.bashrc.
  * a torch_* stage is the lowest to crash (numpy fine)
        -> the torch WHEEL is wrong for this CPU. You have torch 'x.y+cuXXX' — a
           CUDA build — on a Pi with no NVIDIA GPU; its CPU kernels use
           instructions this chip lacks. Install a CPU/ARM build (see footer).
  * only yolo_predict crashes (numpy + torch fine)
        -> the crash is inside ultralytics' torch inference path. Same torch-wheel
           fix; OR sidestep torch entirely by running the NCNN export
           (tests/export_ncnn.py) — NCNN inference does not use torch's kernels.
"""

import argparse
import os
import platform
import signal
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# ── Stages: escalating real computation, each a self-contained snippet ────────
# Ordered low-level -> high-level so the FIRST crash points at the deepest
# broken layer. Each prints "ok" on success; anything else (or a signal) is a
# problem. Keep them small: they must start fast, we run each up to 4 times.

def _stages(model_path: str):
    return [
        (
            "torch_build",
            "report torch build flavour (does NOT crash — just info)",
            "import torch;"
            "print('ver=%s cuda=%s avail=%s'"
            " % (torch.__version__, torch.version.cuda, torch.cuda.is_available()))",
        ),
        (
            "numpy_matmul",
            "256x256 matrix multiply — exercises OpenBLAS",
            "import numpy as np;"
            "a=np.random.rand(256,256).astype('float32');"
            "float((a@a).sum()); print('ok')",
        ),
        (
            "torch_matmul",
            "torch CPU tensor matmul — torch's own BLAS path",
            "import torch;"
            "a=torch.rand(256,256);"
            "float((a@a).sum()); print('ok')",
        ),
        (
            "torch_conv",
            "torch Conv2d forward — the core op every YOLO layer runs",
            "import torch, torch.nn as nn;"
            "m=nn.Conv2d(3,16,3);"
            "float(m(torch.rand(1,3,64,64)).sum()); print('ok')",
        ),
        (
            "yolo_predict",
            f"real YOLO .predict warmup on a dummy frame ({model_path})",
            "import numpy as np;"
            "from ultralytics import YOLO;"
            f"m=YOLO(r'{model_path}');"
            "m.predict(source=np.zeros((320,320,3),dtype='uint8'),"
            "device='cpu', imgsz=320, verbose=False);"
            "print('ok')",
        ),
    ]


def _signal_name(code: int) -> str:
    """returncode is negative when a child is killed by a signal (POSIX)."""
    try:
        return signal.Signals(-code).name
    except (ValueError, AttributeError):
        return f"signal {-code}"


def _run(code: str, armv8: bool) -> tuple[str, str]:
    """Run one snippet in a subprocess. Returns (verdict, detail)."""
    env = os.environ.copy()
    if armv8:
        env["OPENBLAS_CORETYPE"] = "ARMV8"
    else:
        # Make the plain column a fair test: don't inherit an ARMV8 that the
        # user may already have exported, or both columns would look identical.
        env.pop("OPENBLAS_CORETYPE", None)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, cwd=str(ROOT), env=env,
            timeout=180,
        )
    except subprocess.TimeoutExpired:
        return "TIMEOUT", "exceeded 180s"

    if proc.returncode < 0:
        return _signal_name(proc.returncode), f"rc={proc.returncode}"
    if proc.returncode == 0:
        return "OK", proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    last = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "?"
    return "FAIL", last


def _cell(verdict: str) -> str:
    return {"OK": "OK    ", "SIGILL": "SIGILL", "TIMEOUT": "T/O   "}.get(verdict, "FAIL  ")


def main():
    ap = argparse.ArgumentParser(description="Stage-by-stage SIGILL locator")
    ap.add_argument("--model", default=None,
                    help="model to test in the yolo_predict stage "
                         "(default: auto-detect a .pt under src/models)")
    args = ap.parse_args()

    # Auto-detect a model for the final stage if not given.
    model = args.model
    if model is None:
        for cand in ("src/models/best.pt", "src/models/yolov11n-seg.pt",
                     "yolov11n-seg.pt"):
            if (ROOT / cand).exists():
                model = cand
                break
    model = model or "yolov11n-seg.pt"   # let ultralytics try to fetch it

    print("=" * 70)
    print("Runtime SIGILL locator — escalating ops, each in its own process")
    print("-" * 70)
    print(f"  machine  : {platform.machine()}")
    print(f"  python   : {platform.python_version()} ({platform.architecture()[0]})")
    print(f"  model    : {model}  (exists={ (ROOT/model).exists() })")
    print(f"  env OPENBLAS_CORETYPE (inherited): "
          f"{os.environ.get('OPENBLAS_CORETYPE', '<unset>')}")
    print("=" * 70)
    print(f"  {'stage':<14} {'plain':<7} {'+ARMV8':<7} detail")
    print("-" * 70)

    first_crash = None
    armv8_helps = False
    for name, desc, code in _stages(model):
        plain_v, plain_d = _run(code, armv8=False)
        armv8_v, armv8_d = _run(code, armv8=True)
        detail = plain_d if plain_v != "OK" else (armv8_d if armv8_v != "OK" else "")
        print(f"  {name:<14} {_cell(plain_v):<7} {_cell(armv8_v):<7} {detail[:34]}")
        if name == "torch_build":
            continue  # informational only, never a pass/fail stage
        if plain_v != "OK" and first_crash is None:
            first_crash = (name, plain_v)
        if plain_v == "SIGILL" and armv8_v == "OK":
            armv8_helps = True

    print("=" * 70)
    _verdict(first_crash, armv8_helps, model)
    print("=" * 70)


def _verdict(first_crash, armv8_helps, model):
    print("VERDICT")
    print("-" * 70)
    if first_crash is None:
        print("  All runtime stages passed. If the real tests still SIGILL, the")
        print("  crash is in an op not covered here — rerun the test with:")
        print("      OPENBLAS_CORETYPE=ARMV8 python3 tests/test_ibvs_centering.py")
        return

    name, verdict = first_crash
    print(f"  Lowest failing stage: {name}  ({verdict})")
    print()

    if armv8_helps and name in ("numpy_matmul",):
        print("  FIX (confirmed by the +ARMV8 column):")
        print("      echo 'export OPENBLAS_CORETYPE=ARMV8' >> ~/.bashrc && source ~/.bashrc")
        print("  OpenBLAS auto-detected the wrong CPU core; that pins it to ARMv8.")
        return

    if name in ("torch_matmul", "torch_conv", "yolo_predict"):
        print("  This is the torch build. You have a CUDA wheel (+cuXXX) on a Pi with")
        print("  no NVIDIA GPU — its CPU kernels use instructions this chip lacks.")
        print("  Install a CPU/ARM build instead:")
        print("      pip uninstall -y torch torchvision")
        print("      pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
        print("  (If that has no aarch64 wheel for your Python, use the piwheels index")
        print("   or a Pi-specific torch build — tell Claude your exact Python version.)")
        if name == "yolo_predict":
            print()
            print("  FASTEST WORKAROUND (also 2-4x quicker): export to NCNN and let the")
            print("  detector use it — NCNN inference never touches torch's kernels:")
            print(f"      python3 tests/export_ncnn.py --model {model}")
        return

    if name == "numpy_matmul":
        print("  numpy/OpenBLAS crashes and ARMV8 did NOT fix it — reinstall for this arch:")
        print("      pip uninstall -y numpy && pip install numpy")


if __name__ == "__main__":
    main()
