
import os
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from src.utils.config import PROCESSED_DATA_PATH, RANDOM_SEED, TEST_SPLIT
from src.features.pipeline import process_dataset

def load_data():
    print("Loading data...")
    if not os.path.exists(os.path.join(PROCESSED_DATA_PATH, "features.npy")):
        print("Processed data not found. Running pipeline on FULL dataset...")
        process_dataset(max_images_per_class=None, max_classes=None)
        
    try:
        X = np.load(os.path.join(PROCESSED_DATA_PATH, "features.npy"))
        y = np.load(os.path.join(PROCESSED_DATA_PATH, "labels.npy"))
        label_names = np.load(os.path.join(PROCESSED_DATA_PATH, "label_names.npy"), allow_pickle=True)
        print(f"Data Loaded: {X.shape} samples")
        return X, y, label_names
    except FileNotFoundError:
        print("Error loading data even after pipeline run.")
        return None, None, None

def train_models():
    X, y, label_names = load_data()
    if X is None:
        return

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, stratify=y, random_state=RANDOM_SEED
    )
    
    # Define Models
    # Note: LinearSVC doesn't output probabilities by default, so we wrap it or use proper calibration if needed.
    # However, for 'decision_function' based top-3, we might need OneVsRest or CalibratedClassifierCV.
    # We'll use CalibratedClassifierCV for probability output on LinearSVM.
    
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=RANDOM_SEED),
        "KNN": Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=50)), # Reduce dim for KNN speed
            ('knn', KNeighborsClassifier(n_neighbors=5))
        ]),
        "LinearSVM": Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95)), # Keep 95% variance
            ('svm', CalibratedClassifierCV(LinearSVC(dual=False, random_state=RANDOM_SEED)))
        ])
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {acc:.4f}")
        
        results[name] = {
            "model": model,
            "accuracy": acc
        }
        
        # Save Model
        model_path = os.path.join("src/models", f"{name}_model.pkl")
        joblib.dump(model, model_path)
        print(f"Saved {name} to {model_path}")

    # Select Best Model
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    print(f"\nBest Model: {best_name} with {results[best_name]['accuracy']:.4f}")
    
    # Save Label Encoder/Names for inference
    joblib.dump(label_names, os.path.join("src/models", "label_names.pkl"))

if __name__ == "__main__":
    if not os.path.exists("src/models"):
        os.makedirs("src/models")
    train_models()
