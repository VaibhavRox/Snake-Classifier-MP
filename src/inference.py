
"""
Snake Classifier Inference Module.

Supports multiple feature extractors:
- EfficientNet-B0 (1280-dim) - Default
- ResNet50 (2048-dim)
"""

import os
import joblib
import numpy as np
from src.features.extractors import extract_features
from src.utils.safety import check_safety
from src.utils.config import ARTIFACTS_PATH


def _softmax(x):
    """Numerically stable softmax — converts decision scores to pseudo-probabilities."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class SnakeClassifier:
    """
    Loads trained artifacts and runs end-to-end inference.

    Supports multiple feature extractors (EfficientNet, ResNet50).
    """

    def __init__(self, artifacts_dir=None, extractor=None, model_type=None):
        """
        Initialize the classifier.

        Parameters
        ----------
        artifacts_dir : str or None
            Path to artifacts. If None, uses default path.
            If extractor and model_type are provided, appends them to path.
        extractor : str or None
            Feature extractor name (e.g., "efficientnet", "resnet50").
        model_type : str or None
            Model type (e.g., "linearsvc", "logreg", "lgbm").
        """
        # Build artifacts path
        if artifacts_dir:
            self.artifacts_dir = artifacts_dir
        else:
            self.artifacts_dir = ARTIFACTS_PATH

        # Append extractor/model subdirectory if specified
        if extractor and model_type:
            self.artifacts_dir = os.path.join(self.artifacts_dir, extractor, model_type)
        elif extractor:
            self.artifacts_dir = os.path.join(self.artifacts_dir, extractor)

        print(f"Loading artifacts from: {self.artifacts_dir}")

        try:
            self.scaler = joblib.load(os.path.join(self.artifacts_dir, "scaler.pkl"))
            self.model = joblib.load(os.path.join(self.artifacts_dir, "model.pkl"))
            self.label_names = joblib.load(os.path.join(self.artifacts_dir, "label_names.pkl"))

            # Load extractor name if saved
            extractor_path = os.path.join(self.artifacts_dir, "extractor.pkl")
            if os.path.exists(extractor_path):
                self.extractor = joblib.load(extractor_path)
            else:
                # Default to efficientnet if not specified
                self.extractor = extractor or "efficientnet"

            # Load best params if available
            params_path = os.path.join(self.artifacts_dir, "best_params.pkl")
            if os.path.exists(params_path):
                self.best_params = joblib.load(params_path)
            else:
                self.best_params = None

            print(f"Loaded successfully.")
            print(f"  Extractor: {self.extractor}")
            print(f"  Classes: {len(self.label_names)}")
            if self.best_params:
                print(f"  Best params: {self.best_params}")

        except FileNotFoundError as e:
            print(f"Artifact not found: {e}")
            print("Please run training first.")
            self.model = None
            self.extractor = extractor or "efficientnet"

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
            # Extract features using the correct extractor
            features = extract_features(image_path, extractor=self.extractor)
            if features is None:
                return {"error": "Feature extraction failed."}

            # Reshape for sklearn
            features = features.reshape(1, -1)

            # Apply scaler
            features = self.scaler.transform(features)

            # Get predictions
            if hasattr(self.model, "decision_function"):
                scores = self.model.decision_function(features)[0]
                probs = _softmax(scores)
            elif hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(features)[0]
            else:
                return {"error": "Model does not support probability prediction."}

            # Get safety assessment
            top_3, safety_msg = check_safety(probs, self.label_names)

            return {
                "top_3": top_3,
                "safety_message": safety_msg
            }

        except Exception as e:
            return {"error": str(e)}

    def predict_batch(self, image_paths):
        """
        Run inference on multiple images.

        Parameters
        ----------
        image_paths : list
            List of image file paths.

        Returns
        -------
        list
            List of prediction results.
        """
        return [self.predict(path) for path in image_paths]

    def get_raw_scores(self, image_path):
        """
        Get raw prediction scores for an image.

        Returns
        -------
        tuple
            (probabilities, label_names) or (None, None) on error.
        """
        if self.model is None:
            return None, None

        try:
            features = extract_features(image_path, extractor=self.extractor)
            if features is None:
                return None, None

            features = features.reshape(1, -1)
            features = self.scaler.transform(features)

            if hasattr(self.model, "decision_function"):
                scores = self.model.decision_function(features)[0]
                probs = _softmax(scores)
            elif hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(features)[0]
            else:
                return None, None

            return probs, self.label_names

        except Exception:
            return None, None


def load_best_classifier(artifacts_dir=ARTIFACTS_PATH):
    """
    Load the best classifier from the ablation study results.

    Looks for ablation_results.csv to find the best configuration.
    """
    import pandas as pd

    results_path = os.path.join(artifacts_dir, "ablation_results.csv")

    if os.path.exists(results_path):
        results = pd.read_csv(results_path)
        best_row = results.loc[results['Top-1 (%)'].idxmax()]

        extractor = best_row['Extractor'].lower()
        model_type = best_row['Model'].lower()

        print(f"Loading best model: {extractor} + {model_type}")
        print(f"  Top-1: {best_row['Top-1 (%)']}%")

        return SnakeClassifier(
            artifacts_dir=artifacts_dir,
            extractor=extractor,
            model_type=model_type
        )
    else:
        # Fall back to default
        print("No ablation results found. Using default: efficientnet + linearsvc")
        return SnakeClassifier(
            artifacts_dir=artifacts_dir,
            extractor="efficientnet",
            model_type="linearsvc"
        )


if __name__ == "__main__":
    # Try to load the best classifier
    classifier = load_best_classifier()
    print("\nInference system initialized.")
