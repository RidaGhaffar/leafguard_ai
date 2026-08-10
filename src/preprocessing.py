import pandas as pd
import numpy as np

def clean_metadata(df):
    """
    Cleans structural metadata by imputing missing values and removing duplicates.
    """
    df = df.copy()
    df['file_size_kb'] = df['file_size_kb'].fillna(df['file_size_kb'].median())
    df['brightness'] = df['brightness'].fillna(df['brightness'].median())
    if 'crop_type' in df.columns:
        df['crop_type'] = df['crop_type'].fillna(df['crop_type'].mode()[0])
    df = df.drop_duplicates()
    return df

def detect_image_quality(img_np, file_size_kb, brightness, blurriness_score):
    """
    Validates uploaded leaf images against exposure, focus (blurriness), size, and leaf color presence.
    """
    import cv2
    is_valid = True
    errors = []
    
    # 1. Verify if the image is actually a leaf using HSV color checks
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    # Green ranges (healthy leaf)
    lower_green = np.array([35, 20, 20])
    upper_green = np.array([85, 255, 255])
    # Yellow/Brown ranges (diseased leaf spots)
    lower_brown_yellow = np.array([8, 20, 20])
    upper_brown_yellow = np.array([35, 255, 255])
    
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_by = cv2.inRange(hsv, lower_brown_yellow, upper_brown_yellow)
    leaf_mask = cv2.bitwise_or(mask_green, mask_by)
    
    # Percentage of leaf-like pixels
    leaf_pixel_pct = (np.sum(leaf_mask > 0) / leaf_mask.size) * 100
    
    if leaf_pixel_pct < 8.0:
        is_valid = False
        errors.append("Uploaded image does not appear to contain a plant leaf. Please upload a clear photo of a leaf.")
    
    # 2. File size, exposure, and focus checks
    if file_size_kb > 5000:
        is_valid = False
        errors.append(f"File size exceeds 5MB limit ({file_size_kb/1024:.2f}MB). Please compress the image.")
        
    if brightness < 40:
        is_valid = False
        errors.append("Image is underexposed/too dark. Please use better lighting.")
    elif brightness > 220:
        is_valid = False
        errors.append("Image is overexposed/too bright. Please avoid direct flash or glare.")
        
    if blurriness_score < 15:
        is_valid = False
        errors.append("Image is blurry or out of focus. Please capture a sharp photograph.")
        
    return {
        "is_valid": is_valid,
        "errors": errors
    }
