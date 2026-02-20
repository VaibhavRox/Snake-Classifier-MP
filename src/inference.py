
import os
import joblib
import numpy as np
from src.features.extractors import extract_all_features
from src.utils.safety import check_safety
from src.utils.config import ARTIFACTS_PATH


def _softmax(x):
    """Numerically stable softmax — converts decision scores to pseudo-probabilities."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class SnakeClassifier:
    """
    Loads the three separate artifacts (scaler, pca, model) saved by the
    training pipeline and runs end-to-end inference on a single image.
    """

    def __init__(self, artifacts_dir=None):
        self.artifacts_dir = artifacts_dir if artifacts_dir else ARTIFACTS_PATH
        print(f"Loading artifacts from: {self.artifacts_dir}")
        try:
            self.scaler      = joblib.load(os.path.join(self.artifacts_dir, "scaler.pkl"))
            self.pca         = joblib.load(os.path.join(self.artifacts_dir, "pca.pkl"))
            self.model       = joblib.load(os.path.join(self.artifacts_dir, "model.pkl"))
            self.label_names = joblib.load(os.path.join(self.artifacts_dir, "label_names.pkl"))
            print("All artifacts loaded successfully.")
        except FileNotFoundError as e:
            print(f"Artifact not found: {e}\nPlease run training first.")
            self.model = None

    def predict(self, image_path):
        """
        Runs the full inference pipeline on a single image file.

        Returns a dict with keys:
            top_3          — list of {species, probability, is_venomous, venomous_type}
            safety_message — safety assessment string
        or {'error': ...} on failure.
        """
        if self.model is None:
            return {"error": "Model not loaded. Please run training first."}

        try:
            features = extract_all_features(image_path)
            if features is None:
                return {"error": "Feature extraction failed."}

            # Reshape and ensure float32
            features = features.reshape(1, -1).astype(np.float32)

            # Apply the same preprocessing used during training
            features = self.scaler.transform(features)
            features = self.pca.transform(features)

            # Obtain per-class scores → convert to probabilities
            if hasattr(self.model, "decision_function"):
                scores = self.model.decision_function(features)[0]
                probs  = _softmax(scores)
            elif hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(features)[0]
            else:
                return {"error": "Model supports neither decision_function nor predict_proba."}

            top_3, safety_msg = check_safety(probs, self.label_names)

            return {
                "top_3": top_3,
                "safety_message": safety_msg
            }

        except Exception as e:
            return {"error": str(e)}


if __name__ == "__main__":
    classifier = SnakeClassifier()
    print("Inference system initialised.")

