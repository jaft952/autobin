"""
Dataset Structure Checker
Verifies the presence of COCO annotation files and image directories
"""

import os
from pathlib import Path
import json

def check_dataset_structure():
    """Check if all required dataset files exist"""
    
    script_dir = Path(__file__).resolve().parent
    dataset_root = script_dir / '../../data/datasets/raw'
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
def check_labels_match_images(raw_dir):

    annotations_dir = raw_dir / 'annotations'
    images_dir = raw_dir / 'images'
    
    if not images_dir.exists():
        print(f"⚠ images folder not found at {images_dir}")
        return

    for split in ['train', 'val', 'test']:
        json_file = annotations_dir / f"{split}_annotations.coco.json"
        
        if not json_file.exists():
            continue
            
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        images_in_json = data.get('images', [])
        missing_images = []
        
        for img_info in images_in_json:
            file_name = img_info.get('file_name')
            if not (images_dir / file_name).exists():
                missing_images.append(file_name)
                
        print(f"\n[{split}] Images defined in JSON: {len(images_in_json)}")
        if missing_images:
            print(f"  ❌ Missing physical images ({len(missing_images)}):")
            for name in missing_images[:5]:
                print(f"     - {name}")
            if len(missing_images) > 5:
                print(f"     - ... and {len(missing_images) - 5} more")
        else:
            print(f"  ✅ All images defined in {split} JSON physically exist in 'images' folder")

if __name__ == "__main__":
    check_dataset_structure()
    data_yaml = Path(__file__).resolve().parent / '../../data/datasets/raw'
    check_labels_match_images(data_yaml)
