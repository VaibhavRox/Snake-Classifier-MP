"""
Snake Classifier Inference Module.

Uses EfficientNet-B0 features + trained classifier for species identification.
"""

import os
import joblib
import numpy as np
from src.features.extractors import extract_features
from src.utils.safety import check_safety
from src.utils.config import ARTIFACTS_PATH


def _softmax(x):
    """Convert decision scores to probabilities."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class SnakeClassifier:
    """Loads trained model and runs inference."""

    def __init__(self, artifacts_dir=None, model_type="logreg"):
        """
        Initialize classifier.

        Parameters
        ----------
        artifacts_dir : str or None
            Path to artifacts directory.
        model_type : str
            Model type: "linearsvc" or "logreg"
        """
        if artifacts_dir:
            self.artifacts_dir = os.path.join(artifacts_dir, model_type)
        else:
            self.artifacts_dir = os.path.join(ARTIFACTS_PATH, model_type)

        print(f"Loading from: {self.artifacts_dir}")

        try:
            self.scaler = joblib.load(os.path.join(self.artifacts_dir, "scaler.pkl"))
            self.model = joblib.load(os.path.join(self.artifacts_dir, "model.pkl"))
            self.label_names = joblib.load(os.path.join(self.artifacts_dir, "label_names.pkl"))

            params_path = os.path.join(self.artifacts_dir, "best_params.pkl")
            self.best_params = joblib.load(params_path) if os.path.exists(params_path) else None

            print(f"Loaded. Classes: {len(self.label_names)}")

        except FileNotFoundError as e:
            print(f"Error: {e}")
            self.model = None

    def predict(self, image_path):
        """
        Run inference on an image.

        Returns dict with top_3 predictions and safety_message.
        """
        if self.model is None:
            return {"error": "Model not loaded"}

        try:
            features = extract_features(image_path)
            if features is None:
                return {"error": "Feature extraction failed"}

            features = self.scaler.transform(features.reshape(1, -1))

            if hasattr(self.model, "decision_function"):
                scores = self.model.decision_function(features)[0]
                probs = _softmax(scores)
            elif hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(features)[0]
            else:
                return {"error": "Model does not support probabilities"}

            top_3, safety_msg = check_safety(probs, self.label_names)
            return {"top_3": top_3, "safety_message": safety_msg}

        except Exception as e:
            return {"error": str(e)}

    def predict_batch(self, image_paths):
        """Run inference on multiple images."""
        return [self.predict(path) for path in image_paths]


if __name__ == "__main__":
    classifier = SnakeClassifier()
    print("Inference ready.")
