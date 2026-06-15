"""
tests/export_to_onnx.py

Run this ONCE on the Windows training PC (where torch/ultralytics work) to
convert the YOLO .pt model into ONNX. Then copy the resulting .onnx file to the
Raspberry Pi and run tests/test_yolo_model.py there — that test uses cv2.dnn
only, so it needs NO torch / torchvision / ultralytics on the Pi.

    python tests/export_to_onnx.py
"""

from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "ai/models/subsystem2/production/aluminum_can_detector_best.pt"


def main():
    if not MODEL.exists():
        print(f"❌ Model not found: {MODEL}")
        return

    print(f"Loading {MODEL} ...")
    model = YOLO(str(MODEL))

    # opset 12 is safely readable by OpenCV's cv2.dnn. imgsz must match what the
    # Pi-side test feeds the network (640).
    out_path = model.export(format="onnx", imgsz=640, opset=12)
    print(f"✓ Exported ONNX: {out_path}")
    print("Now copy that .onnx next to the .pt on the Pi (same folder) and run:")
    print("    python3 tests/test_yolo_model.py --source can.jpg")


if __name__ == "__main__":
    main()
