
"""
Snake Classifier Inference Module.

Uses VGG16 features + trained classifier for snake species identification.
"""

import os
import joblib
import numpy as np
from src.features.extractors import extract_resnet_features
from src.utils.safety import check_safety
from src.utils.config import ARTIFACTS_PATH


def _softmax(x):
    """Numerically stable softmax — converts decision scores to pseudo-probabilities."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class SnakeClassifier:
    """
    Loads trained artifacts (scaler, model) and runs end-to-end inference
    using VGG16 features.
    """

    def __init__(self, artifacts_dir=None):
        self.artifacts_dir = artifacts_dir if artifacts_dir else ARTIFACTS_PATH
        print(f"Loading artifacts from: {self.artifacts_dir}")

        try:
            self.scaler      = joblib.load(os.path.join(self.artifacts_dir, "scaler.pkl"))
            self.model       = joblib.load(os.path.join(self.artifacts_dir, "model.pkl"))
            self.label_names = joblib.load(os.path.join(self.artifacts_dir, "label_names.pkl"))

            # PCA is optional (we don't use it with VGG)
            pca_path = os.path.join(self.artifacts_dir, "pca.pkl")
            if os.path.exists(pca_path):
                self.pca = joblib.load(pca_path)
                print("Artifacts loaded (with PCA).")
            else:
                self.pca = None
                print("Artifacts loaded (no PCA - using VGG features directly).")

        except FileNotFoundError as e:
            print(f"Artifact not found: {e}\nPlease run training first.")
            self.model = None

    def predict(self, image_path):
        """
        Run inference on a single image file.

        Parameters
        ----------
        image_path : str
            Path to the image file.

        Returns
        -------
        dict
            {
                "top_3": [
                    {"species": str, "probability": float, "is_venomous": bool, "venomous_type": str},
                    ...
                ],
                "safety_message": str
            }
            or {"error": str} on failure.
        """
        if self.model is None:
            return {"error": "Model not loaded. Please run training first."}

        try:
            # Extract VGG16 features
            features = extract_resnet_features(image_path)
            if features is None:
                return {"error": "Feature extraction failed."}

            # Reshape for sklearn
            features = features.reshape(1, -1)

            # Apply scaler
            features = self.scaler.transform(features)

            # Apply PCA if it exists (backward compatibility)
            if self.pca is not None:
                features = self.pca.transform(features)

            # Get predictions
            if hasattr(self.model, "decision_function"):
                scores = self.model.decision_function(features)[0]
                probs  = _softmax(scores)
            elif hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(features)[0]
            else:
                return {"error": "Model supports neither decision_function nor predict_proba."}

            # Get safety assessment
            top_3, safety_msg = check_safety(probs, self.label_names)

            return {
                "top_3": top_3,
                "safety_message": safety_msg
            }

        except Exception as e:
            return {"error": str(e)}


if __name__ == "__main__":
    classifier = SnakeClassifier()
    print("Inference system initialized.")
