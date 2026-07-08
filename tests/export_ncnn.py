"""
tests/export_ncnn.py

One-time (per model) NCNN export for the Raspberry Pi + before/after speed
benchmark. NCNN runs the SAME weights in fp32 — no accuracy change, just an
inference engine that's 2-4x faster on ARM CPUs. After exporting, the
detector picks the NCNN model up AUTOMATICALLY (AluminiumCanDetector prefers
a sibling `<name>_ncnn_model/` directory over the .pt).

Usage (on the Pi, in the repo root):
    python tests/export_ncnn.py                           # default runtime model
    python tests/export_ncnn.py --model src/models/yolov11n-seg.pt
    python tests/export_ncnn.py --bench-only              # just time both

Re-run whenever the .pt is retrained/replaced (the export is a snapshot of
those weights). Requires: pip install ncnn  (ultralytics pulls the rest).
"""
import argparse
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Default = the model CameraSensor actually loads at runtime.
DEFAULT_MODEL = "src/models/best.pt"
IMGSZ = 640
BENCH_RUNS = 10


def bench(model, device, label, imgsz=IMGSZ, runs=BENCH_RUNS):
    import numpy as np
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    model.predict(source=frame, device=device, imgsz=imgsz, verbose=False)  # warmup
    t0 = time.perf_counter()
    for _ in range(runs):
        model.predict(source=frame, device=device, imgsz=imgsz, verbose=False)
    ms = (time.perf_counter() - t0) / runs * 1000.0
    print(f"  {label:<14} {ms:7.1f} ms/frame   (~{1000.0 / ms:.1f} fps)")
    return ms


def main():
    ap = argparse.ArgumentParser(description="Export YOLO .pt to NCNN + benchmark")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=".pt checkpoint path")
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    ap.add_argument("--bench-only", action="store_true",
                    help="skip export, just time .pt vs existing NCNN")
    args = ap.parse_args()

    from pathlib import Path
    from ultralytics import YOLO

    pt_path = Path(args.model)
    if not pt_path.exists():
        print(f"model not found: {pt_path}")
        sys.exit(1)
    ncnn_path = pt_path.with_name(pt_path.stem + "_ncnn_model")

    if not args.bench_only:
        print(f"exporting {pt_path} -> {ncnn_path} (fp32, imgsz={args.imgsz}) ...")
        YOLO(str(pt_path)).export(format="ncnn", imgsz=args.imgsz)
        print("export done.\n")

    if not ncnn_path.is_dir():
        print(f"no NCNN export at {ncnn_path} — run without --bench-only first.")
        sys.exit(1)

    print(f"benchmark ({BENCH_RUNS} runs, imgsz={args.imgsz}, cpu):")
    ms_pt = bench(YOLO(str(pt_path)), "cpu", "PyTorch .pt")
    ms_ncnn = bench(YOLO(str(ncnn_path)), "cpu", "NCNN")
    print(f"\nspeedup: {ms_pt / ms_ncnn:.2f}x — the detector will now use "
          f"{ncnn_path.name} automatically.")


if __name__ == "__main__":
    main()
