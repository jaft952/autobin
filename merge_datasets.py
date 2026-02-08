"""
Merge downloaded datasets with existing raw dataset
Combines: aluminum can detecting + metal cans + existing raw dataset
"""

import json
import shutil
from pathlib import Path
from collections import defaultdict


def merge_datasets():
    # Define paths
    dataset1_dir = Path(r"c:\Users\jaft9\School\autobin\aluminum can detecting.v1i.coco")
    dataset2_dir = Path(r"c:\Users\jaft9\School\autobin\metal cans.v1i.coco")
    raw_dir = Path(r"c:\Users\jaft9\School\autobin\data\datasets\raw")
    
    raw_images_dir = raw_dir / "images"
    raw_annotations_dir = raw_dir / "annotations"
    
    # Create output structure
    raw_images_dir.mkdir(parents=True, exist_ok=True)
    raw_annotations_dir.mkdir(parents=True, exist_ok=True)
    
    # Load existing raw dataset
    existing_anno_file = raw_annotations_dir / "metal_annotations.json"
    with open(existing_anno_file, 'r') as f:
        existing_data = json.load(f)
    
    # For each subset (train, valid, test), we'll merge data
    subsets = ['train', 'valid', 'test']
    merged_data = {subset: {
        'images': [],
        'annotations': [],
        'categories': [],
        'info': None,
        'licenses': []
    } for subset in subsets}
    
    # Track image IDs and annotation IDs for each subset to avoid conflicts
    next_image_id = {subset: 1 for subset in subsets}
    next_annotation_id = {subset: 1 for subset in subsets}
    image_id_mapping = {}  # Old ID -> New ID mapping
    
    print("=" * 60)
    print("MERGING DATASETS")
    print("=" * 60)
    
    # Process Dataset 1: aluminum can detecting
    print("\nProcessing Dataset 1: aluminum can detecting")
    print("-" * 60)
    
    for subset in subsets:
        subset_path = dataset1_dir / subset
        anno_file = subset_path / "_annotations.coco.json"
        
        if not anno_file.exists():
            continue
        
        with open(anno_file, 'r') as f:
            data = json.load(f)
        
        # Copy images
        image_count = 0
        old_to_new_id = {}
        
        for image_info in data['images']:
            old_id = image_info['id']
            image_file = subset_path / image_info['file_name']
            
            if image_file.exists():
                # Copy image
                dst = raw_images_dir / image_info['file_name']
                shutil.copy2(image_file, dst)
                image_count += 1
                
                # Create new image entry with new ID
                new_id = next_image_id[subset]
                old_to_new_id[old_id] = new_id
                
                new_image = image_info.copy()
                new_image['id'] = new_id
                merged_data[subset]['images'].append(new_image)
                next_image_id[subset] += 1
        
        print(f"  {subset}: Copied {image_count} images")
        
        # Copy annotations with updated image IDs
        anno_count = 0
        for anno in data['annotations']:
            if anno['image_id'] in old_to_new_id:
                new_anno = anno.copy()
                new_anno['image_id'] = old_to_new_id[anno['image_id']]
                new_anno['id'] = next_annotation_id[subset]
                merged_data[subset]['annotations'].append(new_anno)
                next_annotation_id[subset] += 1
                anno_count += 1
        
        print(f"         {anno_count} annotations")
        
        # Add categories if not already present
        if not merged_data[subset]['categories']:
            merged_data[subset]['categories'] = data.get('categories', [])
            merged_data[subset]['info'] = data.get('info', {})
            merged_data[subset]['licenses'] = data.get('licenses', [])
    
    # Process Dataset 2: metal cans
    print("\nProcessing Dataset 2: metal cans")
    print("-" * 60)
    
    for subset in subsets:
        subset_path = dataset2_dir / subset
        anno_file = subset_path / "_annotations.coco.json"
        
        if not anno_file.exists():
            continue
        
        with open(anno_file, 'r') as f:
            data = json.load(f)
        
        # Copy images
        image_count = 0
        old_to_new_id = {}
        
        for image_info in data['images']:
            old_id = image_info['id']
            image_file = subset_path / image_info['file_name']
            
            if image_file.exists():
                # Copy image
                dst = raw_images_dir / image_info['file_name']
                shutil.copy2(image_file, dst)
                image_count += 1
                
                # Create new image entry with new ID
                new_id = next_image_id[subset]
                old_to_new_id[old_id] = new_id
                
                new_image = image_info.copy()
                new_image['id'] = new_id
                merged_data[subset]['images'].append(new_image)
                next_image_id[subset] += 1
        
        print(f"  {subset}: Copied {image_count} images")
        
        # Copy annotations with updated image IDs
        anno_count = 0
        for anno in data['annotations']:
            if anno['image_id'] in old_to_new_id:
                new_anno = anno.copy()
                new_anno['image_id'] = old_to_new_id[anno['image_id']]
                new_anno['id'] = next_annotation_id[subset]
                merged_data[subset]['annotations'].append(new_anno)
                next_annotation_id[subset] += 1
                anno_count += 1
        
        print(f"         {anno_count} annotations")
        
        # Merge categories
        if not merged_data[subset]['categories']:
            merged_data[subset]['categories'] = data.get('categories', [])
            merged_data[subset]['info'] = data.get('info', {})
            merged_data[subset]['licenses'] = data.get('licenses', [])
        else:
            # Merge new categories
            existing_names = {cat['name'] for cat in merged_data[subset]['categories']}
            for cat in data.get('categories', []):
                if cat['name'] not in existing_names:
                    merged_data[subset]['categories'].append(cat)
    
    # Add existing raw data (put in train split)
    print("\nProcessing existing raw dataset")
    print("-" * 60)
    
    # The existing data will be added to train
    subset = 'train'
    image_count = 0
    old_to_new_id = {}
    
    for image_info in existing_data['images']:
        old_id = image_info['id']
        image_file = raw_images_dir / image_info['file_name']
        
        if image_file.exists():
            image_count += 1
            new_id = next_image_id[subset]
            old_to_new_id[old_id] = new_id
            
            new_image = image_info.copy()
            new_image['id'] = new_id
            merged_data[subset]['images'].append(new_image)
            next_image_id[subset] += 1
    
    print(f"  Adding {image_count} images to train")
    
    # Copy annotations
    anno_count = 0
    for anno in existing_data['annotations']:
        if anno['image_id'] in old_to_new_id:
            new_anno = anno.copy()
            new_anno['image_id'] = old_to_new_id[anno['image_id']]
            new_anno['id'] = next_annotation_id[subset]
            merged_data[subset]['annotations'].append(new_anno)
            next_annotation_id[subset] += 1
            anno_count += 1
    
    print(f"         {anno_count} annotations")
    
    # Save merged datasets
    print("\nSaving merged annotations...")
    print("-" * 60)
    
    for subset in subsets:
        if merged_data[subset]['images']:
            output_file = raw_annotations_dir / f"{subset}_annotations.coco.json"
            
            # Prepare data
            output_data = {
                'info': merged_data[subset]['info'] or {'description': 'Merged metal datasets'},
                'licenses': merged_data[subset]['licenses'] or [],
                'images': merged_data[subset]['images'],
                'annotations': merged_data[subset]['annotations'],
                'categories': merged_data[subset]['categories']
            }
            
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"  {subset}: {len(merged_data[subset]['images'])} images, "
                  f"{len(merged_data[subset]['annotations'])} annotations")
    
    # Print summary
    print("\n" + "=" * 60)
    print("MERGE SUMMARY")
    print("=" * 60)
    
    total_images = sum(len(merged_data[s]['images']) for s in subsets)
    total_annotations = sum(len(merged_data[s]['annotations']) for s in subsets)
    
    print(f"\nTotal merged:")
    print(f"  Images: {total_images}")
    print(f"  Annotations: {total_annotations}")
    print(f"\nBreakdown:")
    for subset in subsets:
        print(f"  {subset}: {len(merged_data[subset]['images'])} images, "
              f"{len(merged_data[subset]['annotations'])} annotations")
    
    print(f"\nOutput location: {raw_dir}")
    print("=" * 60)


if __name__ == "__main__":
    merge_datasets()
