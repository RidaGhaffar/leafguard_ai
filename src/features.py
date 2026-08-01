import pandas as pd
import numpy as np

def engineer_features(df):
    """
    Applies feature engineering to create resolution, aspect ratio, density score,
    and brightness-blur interaction variables.
    """
    df = df.copy()
    
    # Pixel resolution
    df['resolution'] = df['width'] * df['height']
    
    # Aspect ratio
    df['aspect_ratio'] = df['width'] / df['height']
    
    # Density score (texture detail proxy)
    df['density_score'] = df['file_size_kb'] / df['resolution']
    
    # Interaction metric
    df['brightness_blur_interaction'] = df['brightness'] * df['blurriness_score']
    
    # Handle division by zero or empty values
    df['density_score'] = df['density_score'].fillna(0)
    df['brightness_blur_interaction'] = df['brightness_blur_interaction'].fillna(0)
    
    return df
