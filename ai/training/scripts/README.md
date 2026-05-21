# YOLOv8 Training Scripts

This folder contains all necessary scripts for training a YOLOv8s model to detect aluminum cans.

## Pipeline Overview

```
1. check_dataset.py         → Verify dataset structure
2. convert_coco_to_yolo.py  → Convert COCO to YOLO format  
3. train_yolov8.py          → Train the model
4. monitor_training.py      → Visualize training progress
```

## Step-by-Step Usage

### Step 1: Verify Dataset

```powershell
python training/scripts/check_dataset.py
```

This will check if all COCO annotation files exist and display dataset statistics.

### Step 2: Convert to YOLO Format

```powershell
python training/scripts/convert_coco_to_yolo.py
```

This converts COCO JSON annotations to YOLO txt format and organizes the dataset.

**Output structure:**
```
data/datasets/yolo_format/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

### Step 3: Train the Model

```powershell
python training/scripts/train_yolov8.py
```

**Training Configuration:**
- Model: YOLOv8s (small - balanced speed/accuracy)
- Epochs: 100
- Batch size: 16 (adjust based on GPU memory)
- Image size: 640x640
- Optimizer: SGD with momentum
- Learning rate: 0.01 → 0.0001 (with warmup)
- Augmentations: HSV, rotation, scaling, flips, mosaic, mixup

**Monitor training with TensorBoard:**
```powershell
tensorboard --logdir=runs/detect
```

### Step 4: Visualize Results

```powershell
python training/scripts/monitor_training.py
```

This generates plots of:
- Loss curves (box, class, DFL)
- mAP scores (mAP@0.5, mAP@0.5:0.95)
- Precision and recall
- Learning rate schedule

## Training Output

After training, you'll find:

```
runs/detect/yolov8s_aluminum_can/
├── weights/
│   ├── best.pt          # Best model (highest mAP)
│   └── last.pt          # Last epoch checkpoint
├── results.csv          # Training metrics
├── confusion_matrix.png
├── results.png
├── PR_curve.png
├── F1_curve.png
└── ...
```

The best model is automatically copied to:
```
models/module2/production/aluminum_can_detector_best.pt
```

## Using Pretrained Weights

The training script uses **transfer learning** with YOLOv8s pretrained on COCO dataset. This provides:

✅ Faster convergence (10-20 epochs for good results)  
✅ Better performance with limited data  
✅ Robust feature extraction from general objects  

## Adjusting Hyperparameters

Edit `training/scripts/train_yolov8.py` to modify:

- **Batch size**: Reduce if GPU memory is insufficient
- **Image size**: 416, 512, 640, 800 (larger = more accurate but slower)
- **Epochs**: Increase for more training time
- **Learning rate**: Adjust if loss doesn't converge
- **Augmentation intensity**: Modify hsv_h, degrees, scale values

## GPU Requirements

- **Recommended**: NVIDIA GPU with 6GB+ VRAM
- **Minimum**: 4GB VRAM (reduce batch size to 8)
- **CPU**: Possible but ~10x slower

## Troubleshooting

**Out of memory error:**
```python
batch_size=8  # or 4
```

**Training not converging:**
```python
lr0=0.005     # Lower learning rate
patience=30   # More patience for early stopping
```

**Poor accuracy:**
- Check dataset quality and annotations
- Increase epochs (150-200)
- Add more training data
- Adjust augmentation settings

## Next Steps

After training, use the model for:
1. Real-time detection with webcam
2. Batch inference on test images
3. Integration with robotic arm system
4. Model optimization (ONNX export for deployment)

See `../README.md` for full project documentation.
