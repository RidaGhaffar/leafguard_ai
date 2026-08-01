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

def detect_image_quality(file_size_kb, brightness, blurriness_score):
    """
    Validates uploaded leaf images against exposure, focus (blurriness), and size thresholds
    derived from the IQR outlier boundaries.
    """
    # Thresholds based on IQR outlier calculations:
    # Brightness normal range: [40, 220]
    # Blurriness minimum score: 15 (variance of Laplacian)
    # Max file size: 5000 KB (5 MB)
    
    is_valid = True
    errors = []
    
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
