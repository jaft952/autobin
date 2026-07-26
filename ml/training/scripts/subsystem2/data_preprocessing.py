"""
Data Preprocessing and Augmentation Pipeline
Defines augmentation strategies for training and validation
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np


def get_training_augmentation(img_size=640):
    """
    Get training augmentation pipeline
    
    Args:
        img_size: Target image size for training
        
    Returns:
        Albumentations transform pipeline
    """
    
    transform = A.Compose([
        # Geometric transformations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=20, p=0.5),
        A.Affine(
            shear=(-15, 15),
            p=0.3
        ),
        
        # Scaling and perspective
        A.RandomScale(scale_limit=0.3, p=0.5),
        A.Perspective(scale=(0.05, 0.1), p=0.3),
        
        # Color augmentations
        A.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1,
            p=0.6
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.CLAHE(clip_limit=4.0, p=0.3),
        
        # Blur and noise
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
        ], p=0.3),
        
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        
        # Weather effects (simulating outdoor conditions)
        A.RandomShadow(p=0.2),
        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.1),
        
        # Image quality
        A.ImageCompression(quality_lower=75, quality_upper=100, p=0.3),
        
        # Cutout/Erasing
        A.CoarseDropout(
            max_holes=8,
            max_height=32,
            max_width=32,
            min_holes=1,
            fill_value=0,
            p=0.3
        ),
        
    ], bbox_params=A.BboxParams(
        format='yolo',
        min_visibility=0.3,
        label_fields=['class_labels']
    ))
    
    return transform


def get_validation_augmentation():
    """
    Get validation augmentation pipeline (minimal/no augmentation)
    
    Returns:
        Albumentations transform pipeline
    """
    
    transform = A.Compose([
        # Only normalization for validation
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels']
    ))
    
    return transform


def preprocess_for_inference(image, img_size=640):
    """
    Preprocess image for inference
    
    Args:
        image: Input image (numpy array or path)
        img_size: Target size
        
    Returns:
        Preprocessed image
    """
    
    if isinstance(image, str):
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize while maintaining aspect ratio
    h, w = image.shape[:2]
    scale = img_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    image = cv2.resize(image, (new_w, new_h))
    
    # Pad to square
    pad_h = img_size - new_h
    pad_w = img_size - new_w
    
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    
    return image


# Example usage
if __name__ == "__main__":
    # Test augmentation pipeline
    train_aug = get_training_augmentation()
    val_aug = get_validation_augmentation()
    
    print("Training augmentations loaded successfully")
    print(f"Number of training transforms: {len(train_aug.transforms)}")
    print("\nValidation augmentations loaded successfully")
    print(f"Number of validation transforms: {len(val_aug.transforms)}")
