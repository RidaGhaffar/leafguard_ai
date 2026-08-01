import os
import joblib
import pandas as pd
import numpy as np
from src.features import engineer_features

class LeafGuardPredictor:
    def __init__(self, models_dir="models"):
        """
        Loads the pre-trained Random Forest model, scaler, and expected feature columns template.
        """
        self.model = joblib.load(os.path.join(models_dir, "leafguard_model.joblib"))
        self.scaler = joblib.load(os.path.join(models_dir, "scaler.joblib"))
        self.feature_names = joblib.load(os.path.join(models_dir, "feature_names.joblib"))
        self.scale_cols = ['file_size_kb', 'brightness', 'blurriness_score', 'resolution', 'aspect_ratio', 'density_score', 'brightness_blur_interaction']

    def predict(self, input_data):
        """
        Runs the inference pipeline for a single leaf image sample input.
        input_data: dict containing keys: crop_type, file_size_kb, width, height, brightness, blurriness_score
        """
        # 1. Create DataFrame
        df = pd.DataFrame([input_data])
        
        # 2. Extract Features
        df = engineer_features(df)
        
        # 3. Drop collinear feature 'height'
        df_selected = df.drop(columns=['height'])
        
        # 4. One-Hot Encoding
        df_encoded = pd.get_dummies(df_selected, columns=['crop_type'])
        
        # 5. Align with training feature names (reindex and fill missing one-hot dummies with 0)
        df_aligned = df_encoded.reindex(columns=self.feature_names, fill_value=0)
        
        # 6. Apply StandardScaler scaling
        df_aligned[self.scale_cols] = self.scaler.transform(df_aligned[self.scale_cols])
        
        # 7. Model Inference
        pred_class = self.model.predict(df_aligned)[0] # 1=Healthy, 0=Diseased
        probabilities = self.model.predict_proba(df_aligned)[0]
        
        # Mapping
        status = "Healthy" if pred_class == 1 else "Diseased"
        confidence = float(probabilities[pred_class])
        
        # Recommendations database
        recommendations = {
            "Tomato": {
                "Diseased": "Immediate action: Spray copper fungicide to combat late blight; prune infected bottom leaves.",
                "Healthy": "Condition optimal. Maintain weekly watering, monitor soil humidity, and apply calcium fertilizer."
            },
            "Potato": {
                "Diseased": "Suspected early/late blight. Apply chlorothalonil fungicide and destroy infected foliage.",
                "Healthy": "Healthy crop. Practice crop rotation and monitor soil moisture."
            },
            "Apple": {
                "Diseased": "Black Rot/Rust detected. Prune cankers and apply sulfur-based spray.",
                "Healthy": "Healthy leaf. Maintain normal orchard pruning and inspect regularly."
            },
            "Grape": {
                "Diseased": "Fungal Black Rot. Keep vines pruned for sunlight aeration; spray protective copper.",
                "Healthy": "Vibrant leaf. Inspect leaf undersides weekly for mildew symptoms."
            },
            "Corn": {
                "Diseased": "Common Rust spots. Plant resistant hybrids and spray triazole fungicides if spread exceeds 10%.",
                "Healthy": "Healthy stalk. Ensure nitrogen balance and keep field cleared of weeds."
            },
            "Pepper": {
                "Diseased": "Bacterial Spot. Avoid overhead watering to limit leaf moisture; apply copper bactericide.",
                "Healthy": "Optimal condition. Monitor for aphids and maintain normal watering."
            },
            "Cherry": {
                "Diseased": "Powdery Mildew. Apply sulfur spray early in the morning; prune dense center branches.",
                "Healthy": "Healthy foliage. Spray standard winter dormant spray next season."
            },
            "Peach": {
                "Diseased": "Bacterial Spot leaf damage. Apply copper spray at leaf fall; ensure proper nitrogen fertilization.",
                "Healthy": "Healthy. Monitor for peach tree borer and prune weekly."
            },
            "Strawberry": {
                "Diseased": "Leaf Scorch. Remove dead leaves; treat with chlorothalonil before fruiting.",
                "Healthy": "Vigorous growth. Clean old runners and apply organic straw mulch."
            }
        }
        
        crop = input_data.get("crop_type", "Tomato")
        rec = recommendations.get(crop, {}).get(status, "Inspect leaf regularly for spots or curling.")
        
        return {
            "status": status,
            "confidence": confidence,
            "recommendation": rec
        }
