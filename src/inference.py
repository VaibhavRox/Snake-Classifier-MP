
import os
import joblib
import cv2
import numpy as np
from src.features.extractors import extract_all_features
from src.utils.safety import check_safety
from src.utils.config import IMG_SIZE

MODEL_PATH = "src/models/LinearSVM_model.pkl" # Default to LinearSVM as per plan
LABEL_PATH = "src/models/label_names.pkl"

class SnakeClassifier:
    def __init__(self, model_path=None):
        self.model_path = model_path if model_path else MODEL_PATH
        print(f"Loading model from {self.model_path}...")
        try:
            self.model = joblib.load(self.model_path)
            self.label_names = joblib.load(LABEL_PATH)
            print("Model and labels loaded successfully.")
        except FileNotFoundError:
            print("Model files not found. Please train models first.")
            self.model = None

    def predict(self, image_path):
        if self.model is None:
            return {"error": "Model not loaded"}

        # Extract features
        try:
            features = extract_all_features(image_path)
            if features is None:
                return {"error": "Feature extraction failed"}
            
            # Reshape for single sample
            features = features.reshape(1, -1)
            
            # Get probabilities
            # Ensure model supports predict_proba
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(features)[0]
            else:
                # Fallback for models without proba (shouldn't happen with CalibratedCV)
                return {"error": "Model does not support probability output"}
            
            # Safety Logic
            top_3, safety_msg = check_safety(probs, self.label_names)
            
            return {
                "top_3": top_3,
                "safety_message": safety_msg
            }
            
        except Exception as e:
            return {"error": str(e)}

# For testing functionality
if __name__ == "__main__":
    classifier = SnakeClassifier()
    # Test on a dummy image if exists, or just print status
    print("Inference system initialized.")
