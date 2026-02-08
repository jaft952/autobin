"""
Extract metal category images and annotations from raw dataset
Extract metal (aluminum cans) dataset from combined raw data
"""

import json
import shutil
from pathlib import Path


def extract_metal_dataset():
    # Define paths
    raw_data_dir = Path(r"c:\Users\jaft9\School\autobin\AIAssignment\datasets\raw_data")
    images_dir = raw_data_dir / "combined_images"
    annotations_file = raw_data_dir / "combined_annotations_cleaned.coco.json"
    
    output_dir = Path(r"c:\Users\jaft9\School\autobin\data\datasets\raw")
    output_images_dir = output_dir / "images"
    output_annotations_dir = output_dir / "annotations"
    
    # Create output directories
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_annotations_dir.mkdir(parents=True, exist_ok=True)
    
    # Load original annotations
    print("Loading annotations...")
    with open(annotations_file, 'r') as f:
        original_data = json.load(f)
    
    # Find metal category ID
    metal_category_id = None
    for category in original_data['categories']:
        if category['name'] == 'metal':
            metal_category_id = category['id']
            break
    
    if metal_category_id is None:
        print("ERROR: Metal category not found!")
        return
    
    print(f"Found metal category with ID: {metal_category_id}")
    
    # Filter annotations for metal category
    metal_annotation_ids = set()
    metal_annotations = []
    
    for annotation in original_data['annotations']:
        if annotation['category_id'] == metal_category_id:
            metal_annotation_ids.add(annotation['image_id'])
            metal_annotations.append(annotation)
    
    print(f"Found {len(metal_annotations)} metal annotations")
    
    # Filter images that have metal annotations
    metal_images = []
    for image in original_data['images']:
        if image['id'] in metal_annotation_ids:
            metal_images.append(image)
    
    print(f"Found {len(metal_images)} images with metal objects")
    
    # Copy metal images
    print("\nCopying images...")
    copied_count = 0
    for image in metal_images:
        src_image = images_dir / image['file_name']
        dst_image = output_images_dir / image['file_name']
        
        if src_image.exists():
            shutil.copy2(src_image, dst_image)
            copied_count += 1
        else:
            print(f"WARNING: Image not found: {src_image}")
    
    print(f"Successfully copied {copied_count} images")
    
    # Create new COCO format JSON with only metal data
    print("\nCreating annotations file...")
    metal_coco_data = {
        'info': original_data['info'],
        'licenses': original_data['licenses'],
        'categories': [cat for cat in original_data['categories'] if cat['id'] == metal_category_id],
        'images': metal_images,
        'annotations': metal_annotations
    }
    
    # Save new annotations file
    output_annotations_file = output_annotations_dir / "metal_annotations.json"
    with open(output_annotations_file, 'w') as f:
        json.dump(metal_coco_data, f, indent=2)
    
    print(f"Saved annotations to: {output_annotations_file}")
    
    # Print summary
    print("\n" + "="*50)
    print("EXTRACTION SUMMARY")
    print("="*50)
    print(f"Total metal images: {len(metal_images)}")
    print(f"Total metal annotations: {len(metal_annotations)}")
    print(f"Images copied: {copied_count}")
    print(f"Output directory: {output_dir}")
    print(f"Images saved to: {output_images_dir}")
    print(f"Annotations saved to: {output_annotations_file}")
    print("="*50)


if __name__ == "__main__":
    extract_metal_dataset()
