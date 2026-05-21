# YOLOv8 Aluminum Can Detection Training - Setup Guide

## Overview

This guide will help you set up the environment and train a YOLOv8 model to detect aluminum cans for the robotic arm waste collection system (Module 2).

## Prerequisites

- **Anaconda/Miniconda** installed on your system
- **VS Code** with Python and Jupyter extensions installed
- **NVIDIA GPU** (recommended for faster training, optional)

## Installation Steps

### Step 1: Create a Conda Environment

Open your terminal/PowerShell and create a dedicated environment for this project:

```bash
conda create -n fyp python=3.10 -y
conda activate fyp
```

**Explanation:**
- `-n fyp`: Environment name
- `python=3.10`: Python version (compatible with YOLOv8)
- `-y`: Auto-confirm installation

### Step 2: Install Dependencies

Navigate to your project directory and install all required packages:

```bash
cd c:\Users\jaft9\School\autobin

pip install -r requirements.txt
```

**What gets installed:**
- `torch` & `torchvision`: PyTorch deep learning framework
- `ultralytics`: YOLOv8 library
- `opencv-python`: Image processing
- `numpy`, `pandas`: Data handling
- `matplotlib`, `tensorboard`: Visualization and logging

**For GPU Support (NVIDIA only):**
If you have an NVIDIA GPU, install CUDA-enabled PyTorch:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 3: Configure VS Code

1. Open VS Code (with your project folder)
2. Install these extensions:
   - **Python** (by Microsoft)
   - **Jupyter** (by Microsoft)
   - **Python Environment Manager** (by Microsoft) - optional but helpful

3. Select the kernel for your notebook:
   - Open `training/scripts/module1.ipynb`
   - Click "Select Kernel" (top right)
   - Choose "Python Environments"
   - Select `yolov8_aluminum`

### Step 4: Verify Installation

Run this in your notebook or terminal to verify everything works:

```python
import torch
import ultralytics
from ultralytics import YOLO

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Ultralytics version: {ultralytics.__version__}")
```

Expected output:
```
PyTorch version: 2.x.x
CUDA available: True (or False if no GPU)
Ultralytics version: 8.x.x
```

## Dataset Structure

Your training dataset is already prepared in:
```
data/datasets/raw/
├── images/
│   └── [1564 aluminum can images]
├── annotations/
│   ├── train_annotations.coco.json     (1506 images, 1552 annotations)
│   ├── val_annotations.coco.json       (287 images, 293 annotations)
│   └── test_annotations.coco.json      (184 images, 186 annotations)
```

**Dataset Stats:**
- **Total images:** 1,977
- **Total annotations:** 2,031
- **Class:** `aluminium_can` (single class)
- **Format:** COCO JSON

## Training YOLOv8 Model

### Quick Start Example

```python
from ultralytics import YOLO

# Load a pretrained YOLOv8n (nano) model
model = YOLO('yolov8n.pt')

# Train the model
results = model.train(
    data='path/to/data.yaml',  # YOLO format dataset config
    epochs=50,
    imgsz=640,
    device=0,  # GPU device (0 for first GPU, -1 for CPU)
    patience=20,  # Early stopping patience
    save=True,
    project='runs/aluminum_can_detection',
    name='yolov8n_v1'
)

# Validate
metrics = model.val()

# Test on single image
results = model.predict(source='path/to/test/image.jpg', conf=0.25)

# Export model
model.export(format='onnx')  # or 'tflite', 'torchscript', etc.
```

### Converting COCO to YOLO Format (if needed)

YOLOv8 requires a `data.yaml` file. Example:

```yaml
path: /absolute/path/to/data/datasets/raw
train: images  # train images folder
val: images    # val images folder  
test: images   # test images folder

nc: 1  # number of classes
names: ['aluminium_can']  # class names
```

## Expected Performance Targets

Based on your requirements:
- **Detection Accuracy (mAP@0.5):** ≥80%
- **Inference Speed:** <200ms per frame on GPU
- **False Positive Rate:** <5%
- **Recall:** ≥75% for reliable robotic picking

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "ModuleNotFoundError: No module named 'torch'" | Run `pip install -r requirements.txt` |
| Kernel not showing up in VS Code | Restart VS Code, ensure Python extension is installed |
| GPU not detected | Install `torch` with CUDA: see Step 2 |
| Out of memory during training | Reduce `batch_size` in training config |
| COCO annotation format not recognized | Convert to YOLO format with data.yaml |

## Next Steps

1. ✅ Environment setup (this guide)
2. Convert COCO annotations to YOLO format (if using YOLOv8 native training)
3. Create `training/scripts/train_yolov8.ipynb` to run training
4. Evaluate model performance on validation/test sets
5. Export trained model for robotic arm deployment

## Additional Resources

- **YOLOv8 Docs:** https://docs.ultralytics.com/
- **YOLOv8 GitHub:** https://github.com/ultralytics/ultralytics
- **PyTorch Docs:** https://pytorch.org/docs/stable/index.html

## Questions?

Refer to your requirements document: `docs/requirementDocs.txt`

---

## Project Folder Structure

```
autobin/
├── .git/                           # Git version control
├── .gitignore                      # Git ignore patterns
│
├── data/                           # Dataset storage
│   └── datasets/
│       ├── module1/                # Module 1 datasets (if any)
│       ├── module2/                # Module 2 datasets (if any)
│       ├── raw/                    # Raw aluminum can detection dataset
│       │   ├── annotations/
│       │   │   ├── train_annotations.coco.json  (1506 images, 1552 annotations)
│       │   │   ├── val_annotations.coco.json    (287 images, 293 annotations)
│       │   │   └── test_annotations.coco.json   (184 images, 186 annotations)
│       │   └── images/             # 1,977 aluminum can images
│       └── yolo_format/            # Converted YOLO format
│           └── data.yaml           # YOLOv8 dataset configuration
│
├── docs/                           # Documentation
│   ├── FYPSRS.pdf                  # Project requirements document (PDF)
│   └── requirementDocs.txt         # Requirements specification
│
├── logs/                           # Training and runtime logs
│   ├── detection/                  # Detection inference logs
│   ├── performance/                # Performance metrics logs
│   └── training/                   # Training logs
│
├── models/                         # Trained model storage
│   ├── module1/                    # Module 1 (aerial interception) models
│   │   ├── checkpoints/            # Training checkpoints
│   │   ├── experiments/            # Experimental models
│   │   └── production/             # Production-ready models
│   └── module2/                    # Module 2 (ground collection) models
│       ├── checkpoints/            # Training checkpoints
│       ├── experiments/            # Experimental models
│       └── production/             # Production-ready models
│
├── runs/                           # YOLOv8 training runs output (empty initially)
│
├── src/                            # Source code
│
├── tests/                          # Unit and integration tests
│
├── training/                       # Training scripts and configs
│   ├── configs/                    # Training configuration files
│   ├── results/                    # Training results and metrics
│   ├── scripts/                    # Training notebooks and scripts
│   │   ├── check_dataset.py        # Dataset validation script
│   │   ├── convert_coco_to_yolo.py # COCO to YOLO conversion
│   │   ├── module1.ipynb           # Module 1 training notebook
│   │   ├── monitor_training.py     # Training monitoring utilities
│   │   ├── train_yolov8.py         # YOLOv8 training script
│   │   └── README.md               # Training scripts documentation
│   └── utils/                      # Training utility functions
│
├── extract_metal_dataset.py        # Script to extract metal/aluminum data
├── merge_datasets.py               # Script to merge multiple datasets
├── README.md                       # This file (main project documentation)
├── requirements.txt                # Python dependencies
└── setup.py                        # Package installation script
```

### Key Directories Explained

- **`data/datasets/raw/`**: Contains the aluminum can detection dataset in COCO format with 1,977 images and 2,031 annotations across train/val/test splits. All categories unified to single class `aluminium_can`.

- **`training/scripts/`**: Training notebooks and Python scripts for YOLOv8 model training, including dataset conversion utilities.

- **`models/module2/`**: Storage for trained aluminum can detection models for the robotic arm (Module 2: Ground Trash Collection).

- **`logs/`**: Organized logs for training runs, detection inference, and performance metrics tracking.

- **`runs/`**: YOLOv8's default output directory for training runs (populated after training begins).
