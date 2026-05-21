"""
COCO to YOLO Format Converter
Converts COCO JSON annotations to YOLO txt format for YOLOv8 training
"""

import json
import os
import shutil
from pathlib import Path
from tqdm import tqdm

def convert_coco_to_yolo(coco_json_path, output_labels_dir, images_src_dir, images_dst_dir):
    """
    Convert COCO format to YOLOv8 format
    
    Args:
        coco_json_path: Path to COCO JSON file
        output_labels_dir: Directory to save YOLO label files
        images_src_dir: Source directory containing images
        images_dst_dir: Destination directory to copy images
    """
    
    print(f"\nProcessing: {coco_json_path}")
    
    # Load COCO data
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create output directories
    os.makedirs(output_labels_dir, exist_ok=True)
    os.makedirs(images_dst_dir, exist_ok=True)
    
    # Create image info dictionary
    image_sizes = {}
    for img in coco_data['images']:
        image_sizes[img['id']] = {
            'width': img['width'],
            'height': img['height'],
            'file_name': img['file_name']
        }
    
    # Create category mapping (COCO categories start at 1, YOLO at 0)
    category_map = {}
    for cat in coco_data['categories']:
        category_map[cat['id']] = cat['name']
    
    print(f"Categories found: {category_map}")
    
    # Group annotations by image
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    # Convert and save YOLO format labels
    converted_count = 0
    copied_images = 0
    
    for img_id, anns in tqdm(annotations_by_image.items(), desc="Converting annotations"):
        img_info = image_sizes.get(img_id)
        if not img_info:
            continue
            
        img_width = img_info['width']
        img_height = img_info['height']
        file_name = img_info['file_name']
        
        # Create label file path
        label_file = Path(output_labels_dir) / (Path(file_name).stem + '.txt')
        
        # Write YOLO format annotations
        with open(label_file, 'w') as f:
            for ann in anns:
                cat_id = ann['category_id']
                bbox = ann['bbox']  # [x, y, width, height] in pixels
                
                # Convert to YOLO format: [class_id, center_x, center_y, width, height] normalized
                x, y, w, h = bbox
                center_x = (x + w / 2) / img_width
                center_y = (y + h / 2) / img_height
                norm_width = w / img_width
                norm_height = h / img_height
                
                # YOLO class_id starts from 0
                class_id = cat_id - 1 if cat_id > 0 else cat_id
                
                # Write to file
                f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}\n")
        
        converted_count += 1
        
        # Copy corresponding image
        src_image = Path(images_src_dir) / file_name
        dst_image = Path(images_dst_dir) / file_name
        
        if src_image.exists():
            shutil.copy2(src_image, dst_image)
            copied_images += 1
        else:
            print(f"Warning: Image not found: {src_image}")
    
    print(f"✓ Converted {converted_count} label files")
    print(f"✓ Copied {copied_images} images")
    print(f"  Labels saved to: {output_labels_dir}")
    print(f"  Images saved to: {images_dst_dir}")


def main():
    """Convert all COCO annotations to YOLO format"""
    
    print("=" * 60)
    print("COCO to YOLO Format Conversion")
    print("=" * 60)
    
    # Define paths
    base_dir = Path("../../data/datasets")
    raw_dir = base_dir / "raw"
    yolo_dir = base_dir / "yolo_format"
    
    splits = ["train", "val", "test"]
    
    for split in splits:
        coco_json = raw_dir / "annotations" / f"{split}_annotations.coco.json"
        labels_output = yolo_dir / "labels" / split
        images_src = raw_dir / "images"
        images_dst = yolo_dir / "images" / split
        
        if coco_json.exists():
            convert_coco_to_yolo(
                coco_json_path=str(coco_json),
                output_labels_dir=str(labels_output),
                images_src_dir=str(images_src),
                images_dst_dir=str(images_dst)
            )
        else:
            print(f"⚠ Skipping {split}: {coco_json} not found")
    
    print("\n" + "=" * 60)
    print("Conversion Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
