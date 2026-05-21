"""
Dataset Structure Checker
Verifies the presence of COCO annotation files and image directories
"""

import os
from pathlib import Path
import json

def check_dataset_structure():
    """Check if all required dataset files exist"""
    
    dataset_root = Path("../../data/datasets/raw")
    annotations_dir = dataset_root / "annotations"
    images_dir = dataset_root / "images"
    
    print("=" * 60)
    print("Dataset Structure Check")
    print("=" * 60)
    
    # Check annotation files
    print("\n[Annotation Files]")
    for split in ["train", "val", "test"]:
        coco_file = annotations_dir / f"{split}_annotations.coco.json"
        
        if coco_file.exists():
            # Load and print stats
            with open(coco_file, 'r') as f:
                data = json.load(f)
                num_images = len(data.get('images', []))
                num_annotations = len(data.get('annotations', []))
                num_categories = len(data.get('categories', []))
                
            print(f"✓ {split}_annotations.coco.json")
            print(f"  - Images: {num_images}")
            print(f"  - Annotations: {num_annotations}")
            print(f"  - Categories: {num_categories}")
            
            # Print category names
            if num_categories > 0:
                categories = data.get('categories', [])
                cat_names = [cat['name'] for cat in categories]
                print(f"  - Category names: {cat_names}")
        else:
            print(f"✗ Missing {split}_annotations.coco.json")
    
    # Check images directory
    print(f"\n[Images Directory]")
    if images_dir.exists():
        print(f"✓ Images directory exists: {images_dir}")
        
        # Count image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(images_dir.glob(f"**/*{ext}")))
        
        print(f"  - Total images found: {len(image_files)}")
    else:
        print(f"✗ Images directory not found: {images_dir}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    check_dataset_structure()
