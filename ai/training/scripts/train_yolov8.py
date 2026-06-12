"""
YOLOv8 Training Script
Train YOLOv8 models for the autobin perception stack.

Presets (pick with --preset):
  can       single-class aluminum can detector (original behavior, default)
  combined  2-class detector: aluminum_can + gripper, ONE model for closed-loop
            visual servoing (one inference pass returns both boxes per frame).

Examples:
  python train_yolov8.py                       # original can training
  python train_yolov8.py --preset combined     # can + gripper 2-class training
  python train_yolov8.py --preset combined --model-size n --epochs 150
"""

import argparse
from ultralytics import YOLO
import torch
from pathlib import Path
import yaml


# Per-preset configuration. data_yaml paths are relative to this script's folder.
PRESETS = {
    "can": {
        "data_yaml": "../../data/datasets/yolo_format/data.yaml",
        "run_name": "yolov8s_aluminum_can",
        "single_cls": True,
        "production_name": "aluminum_can_detector_best.pt",
    },
    "combined": {
        "data_yaml": "../../data/datasets/yolo_format_combined/data.yaml",
        "run_name": "yolov8_can_gripper",
        "single_cls": False,   # CRITICAL: keep classes separate (can vs gripper)
        "production_name": "can_gripper_detector_best.pt",
    },
}


def check_gpu():
    """Check if GPU is available"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ GPU Available: {gpu_name}")
        print(f"  Memory: {gpu_memory:.2f} GB")
        return 0  # Use GPU 0
    else:
        print("⚠ No GPU detected, using CPU (training will be slower)")
        return 'cpu'


def train_yolov8_detector(
    model_size='s',
    epochs=100,
    batch_size=16,
    img_size=640,
    device=None,
    resume=False,
    preset='can',
):
    """
    Train a YOLOv8 detector for the given preset (see PRESETS).

    Args:
        model_size: Model size ('n', 's', 'm', 'l', 'x')
        epochs: Number of training epochs
        batch_size: Batch size
        img_size: Input image size
        device: Device to use (None for auto-detect)
        resume: Resume from last checkpoint
        preset: 'can' (single-class, original) or 'combined' (can + gripper)
    """
    cfg = PRESETS[preset]

    print("=" * 70)
    print(f"YOLOv8 Detector Training — preset '{preset}' ({cfg['run_name']})")
    print("=" * 70)

    # Check device
    if device is None:
        device = check_gpu()

    model_name = f'yolov8{model_size}.pt'
    print(f"\nLoading pretrained model: {model_name}")
    model = YOLO(model_name)

    # Dataset configuration
    data_yaml = Path(__file__).resolve().parent / cfg['data_yaml']

    # Verify data.yaml exists
    if not Path(data_yaml).exists():
        raise FileNotFoundError(f"Dataset configuration not found: {data_yaml}")
    
    print(f"Dataset config: {data_yaml}")
    
    # Training configuration
    print("\n" + "=" * 70)
    print("Training Configuration")
    print("=" * 70)
    print(f"Model: YOLOv8{model_size}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Image Size: {img_size}")
    print(f"Device: {device}")
    print(f"Optimizer: SGD")
    print(f"Initial LR: 0.01")
    print("=" * 70 + "\n")
    
    # Start training
    results = model.train(
        # Data
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        
        # Optimization
        optimizer='SGD',
        lr0=0.01,              # Initial learning rate
        lrf=0.01,              # Final learning rate (lr0 * lrf)
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # Augmentation (built-in YOLOv8 augmentations)
        hsv_h=0.015,           # HSV-Hue augmentation
        hsv_s=0.7,             # HSV-Saturation augmentation  
        hsv_v=0.4,             # HSV-Value augmentation
        degrees=10.0,          # Rotation (+/- deg)
        translate=0.1,         # Translation (+/- fraction)
        scale=0.5,             # Scaling (+/- gain)
        shear=5.0,             # Shear (+/- deg)
        perspective=0.0005,    # Perspective
        flipud=0.0,            # Vertical flip probability
        fliplr=0.5,            # Horizontal flip probability
        mosaic=1.0,            # Mosaic augmentation probability
        mixup=0.1,             # Mixup augmentation probability
        copy_paste=0.0,        # Copy-paste augmentation probability
        
        # Loss weights
        box=7.5,               # Box loss weight
        cls=0.5,               # Classification loss weight
        dfl=1.5,               # Distribution focal loss weight
        
        # Validation & Checkpointing
        val=True,
        patience=20,           # Early stopping patience
        save=True,
        save_period=10,        # Save checkpoint every N epochs
        
        # Directories
        project='runs/detect',
        name=cfg['run_name'],
        exist_ok=False,

        # Other
        pretrained=True,
        verbose=True,
        seed=42,
        deterministic=True,
        single_cls=cfg['single_cls'],  # False for multi-class (can + gripper)
        rect=False,            # Rectangular training
        cos_lr=False,          # Cosine learning rate scheduler
        close_mosaic=10,       # Disable mosaic in final N epochs
        resume=resume,         # Resume training
        amp=True,              # Automatic Mixed Precision
        fraction=1.0,          # Dataset fraction to train on
        
        # Visualization
        plots=True,
    )
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    
    return model, results


def evaluate_model(model, data_yaml):
    """
    Evaluate trained model on validation set
    
    Args:
        model: Trained YOLO model
        data_yaml: Path to dataset configuration
    """
    
    print("\n" + "=" * 70)
    print("Model Evaluation")
    print("=" * 70)
    
    metrics = model.val(data=data_yaml, split='test')
    
    print(f"\nmAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    
    return metrics


def export_model(model, export_format='onnx'):
    """
    Export model to different formats
    
    Args:
        model: Trained YOLO model
        export_format: Export format ('onnx', 'torchscript', 'tflite', etc.)
    """
    
    print(f"\nExporting model to {export_format} format...")
    model.export(format=export_format)
    print("✓ Export complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train autobin YOLOv8 detectors")
    parser.add_argument('--preset', choices=list(PRESETS), default='can',
                        help="'can' = original 1-class detector, "
                             "'combined' = 2-class can + gripper detector")
    parser.add_argument('--model-size', default='s', choices=['n', 's', 'm', 'l', 'x'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    cfg = PRESETS[args.preset]

    # Train the model
    model, results = train_yolov8_detector(
        model_size=args.model_size,
        epochs=args.epochs,
        batch_size=args.batch_size,  # Adjust based on GPU memory
        img_size=args.img_size,
        device=None,                 # Auto-detect GPU/CPU
        resume=args.resume,
        preset=args.preset,
    )

    script_dir = Path(__file__).resolve().parent
    root_dir = script_dir.parent.parent
    best_model_path = root_dir / f"runs/detect/{cfg['run_name']}/weights/best.pt"
    data_yaml = str(script_dir / cfg['data_yaml'])
    # Evaluate on test set
    metrics = evaluate_model(model, data_yaml=data_yaml)

    # Save the best model to production directory
    if best_model_path.exists():
        import shutil
        production_dir = root_dir / 'models/subsystem2/production'
        production_dir.mkdir(parents=True, exist_ok=True)

        production_path = production_dir / cfg['production_name']
        shutil.copy2(best_model_path, production_path)
        print(f"\n✓ Best model saved to: {production_path}")


    print("\n" + "=" * 70)
    print("All training tasks completed!")
    print("=" * 70)
