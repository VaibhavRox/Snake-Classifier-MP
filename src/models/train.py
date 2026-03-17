"""
Snake Classifier — Training Pipeline
=====================================
Pipeline order (no data leakage):
    Load → Split (stratified 80/20) → Scale → PCA → Train → Evaluate → Save

Models supported:
    linearsvc  — LinearSVC  (fast, memory-efficient, recommended)
    logreg     — Multinomial Logistic Regression (saga solver)
    lgbm       — LightGBM  (optional; requires lightgbm package)
"""

import os
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, top_k_accuracy_score

from src.utils.config import (
    PROCESSED_DATA_PATH, ARTIFACTS_PATH, RANDOM_SEED, TEST_SPLIT, PCA_COMPONENTS
)
from src.features.pipeline import process_dataset


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load Data
# ─────────────────────────────────────────────────────────────────────────────

def load_data(use_hog=True, use_lbp=True, use_hsv=True):
    """
    Load preprocessed feature matrix and labels.
    If the compressed .npz file doesn't exist, runs the extraction pipeline first.
    Returns (X: float32, y, label_names) or (None, None, None) on failure.
    """
    print("Loading data...")
    data_file = os.path.join(PROCESSED_DATA_PATH, "features.npz")

    if not os.path.exists(data_file):
        # Backward-compatibility: try legacy .npy files
        npy_x = os.path.join(PROCESSED_DATA_PATH, "features.npy")
        if not os.path.exists(npy_x):
            print("Processed data not found — running feature extraction pipeline...")
            process_dataset(
                max_images_per_class=None, max_classes=None,
                use_hog=use_hog, use_lbp=use_lbp, use_hsv=use_hsv
            )

    try:
        if os.path.exists(data_file):
            data = np.load(data_file, allow_pickle=True)
            X           = data["features"].astype(np.float32)
            y           = data["labels"]
            label_names = data["label_names"]
        else:
            # Legacy fallback
            X           = np.load(os.path.join(PROCESSED_DATA_PATH, "features.npy")).astype(np.float32)
            y           = np.load(os.path.join(PROCESSED_DATA_PATH, "labels.npy"))
            label_names = np.load(os.path.join(PROCESSED_DATA_PATH, "label_names.npy"), allow_pickle=True)

        print(f"Loaded: {X.shape} samples  |  dtype: {X.dtype}  |  classes: {len(np.unique(y))}")
        return X, y, label_names

    except FileNotFoundError:
        print("Error: Could not load data even after pipeline run.")
        return None, None, None


# ─────────────────────────────────────────────────────────────────────────────
# 2. Split Data
# ─────────────────────────────────────────────────────────────────────────────

def split_data(X, y):
    """
    Stratified 80/20 split performed BEFORE any scaling or PCA to prevent
    data leakage.
    """
    print(f"Splitting data — stratified {int((1-TEST_SPLIT)*100)}/{int(TEST_SPLIT*100)}...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, stratify=y, random_state=RANDOM_SEED
    )
    print(f"  Train : {X_train.shape}")
    print(f"  Test  : {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────────────────
# 3. Preprocess (Scale → PCA)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(X_train, X_test, n_components=PCA_COMPONENTS):
    """
    1. StandardScaler — fitted ONLY on training data.
    2. PCA            — fitted ONLY on training data.
    Returns transformed arrays plus the fitted scaler and pca objects.
    """
    print("\nScaling features (StandardScaler)...")
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print(f"Applying PCA (n_components={n_components})...")
    pca     = PCA(n_components=n_components, random_state=RANDOM_SEED)
    X_train = pca.fit_transform(X_train)
    X_test  = pca.transform(X_test)

    explained = pca.explained_variance_ratio_.sum() * 100
    print(f"  After PCA — Train: {X_train.shape}  Test: {X_test.shape}")
    print(f"  Explained variance: {explained:.1f}%")

    return X_train, X_test, scaler, pca


# ─────────────────────────────────────────────────────────────────────────────
# 4. Train Model
# ─────────────────────────────────────────────────────────────────────────────

def train_model(X_train, y_train, model_type="linearsvc"):
    """
    Trains the chosen model on (already scaled + PCA-reduced) training data.

    model_type : "linearsvc" | "logreg" | "lgbm"
    """
    model_type = model_type.lower()

    if model_type == "linearsvc":
        print("\nTraining LinearSVC (optimized for multiclass, memory-efficient)...")
        model = LinearSVC(
            C=2.0,
            max_iter=20000,
            verbose=1,
            random_state=RANDOM_SEED,
            class_weight="balanced",
            dual=True,
            tol=1e-4
        )

    elif model_type == "logreg":
        print("\nTraining Logistic Regression (saga)...")
        model = LogisticRegression(
            max_iter=5000,
            solver="saga",
            class_weight="balanced",
            random_state=RANDOM_SEED
        )

    elif model_type == "lgbm":
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError(
                "LightGBM is not installed. Run: pip install lightgbm"
            )
        print("\nTraining LightGBM...")
        model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=30,
            max_depth=5,
            n_jobs=-1,
            class_weight="balanced",
            random_state=RANDOM_SEED
        )

    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Choose 'linearsvc', 'logreg', or 'lgbm'."
        )

    model.fit(X_train, y_train)
    print("Training complete.")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 5. Evaluate Model
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test):
    """
    Computes Top-1 and Top-5 accuracy and prints a summary.
    Returns (top1_acc, top5_acc).
    """
    y_pred = model.predict(X_test)
    top1   = accuracy_score(y_test, y_pred)

    # decision_function / predict_proba for top-k
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
    elif hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_test)
    else:
        scores = None

    if scores is not None:
        top5 = top_k_accuracy_score(y_test, scores, k=5)
    else:
        top5 = float("nan")

    print(f"\n{'='*45}")
    print(f"  Top-1 Accuracy : {top1 * 100:.2f}%")
    print(f"  Top-5 Accuracy : {top5 * 100:.2f}%")
    print(f"{'='*45}\n")

    return top1, top5


# ─────────────────────────────────────────────────────────────────────────────
# 6. Cross Validation (optional)
# ─────────────────────────────────────────────────────────────────────────────

def cross_validate(model_type, X_train, y_train, n_splits=3):
    """
    StratifiedKFold cross-validation on training data.
    Uses fresh (unfitted) model instances to avoid leakage.
    """
    print(f"\nRunning {n_splits}-fold StratifiedKFold CV ({model_type})...")

    model_type = model_type.lower()
    if model_type == "linearsvc":
        cv_model = LinearSVC(
            C=1.0,
            max_iter=10000,
            verbose=1,
            random_state=RANDOM_SEED,
            class_weight="balanced",
            dual=False,
            tol=1e-3
        )
    elif model_type == "logreg":
        cv_model = LogisticRegression(
            max_iter=1000,
            solver="saga",
            class_weight="balanced",
            random_state=RANDOM_SEED
        )
    elif model_type == "lgbm":
        try:
            import lightgbm as lgb
            cv_model = lgb.LGBMClassifier(
                n_estimators=500, learning_rate=0.05, num_leaves=63,
                n_jobs=-1, class_weight="balanced", random_state=RANDOM_SEED
            )
        except ImportError:
            print("LightGBM not installed — skipping CV.")
            return None
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    skf    = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_val_score(cv_model, X_train, y_train, cv=skf, scoring="accuracy", n_jobs=-1)

    print(f"  CV Scores      : {[f'{s*100:.2f}%' for s in scores]}")
    print(f"  Mean ± Std     : {scores.mean()*100:.2f}% ± {scores.std()*100:.2f}%")
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# 7. Save Artifacts
# ─────────────────────────────────────────────────────────────────────────────

def save_artifacts(scaler, pca, model, label_names, artifacts_dir=ARTIFACTS_PATH):
    """
    Saves scaler.pkl, pca.pkl, model.pkl, label_names.pkl via joblib.
    """
    os.makedirs(artifacts_dir, exist_ok=True)

    joblib.dump(scaler,      os.path.join(artifacts_dir, "scaler.pkl"))
    joblib.dump(pca,         os.path.join(artifacts_dir, "pca.pkl"))
    joblib.dump(model,       os.path.join(artifacts_dir, "model.pkl"))
    joblib.dump(label_names, os.path.join(artifacts_dir, "label_names.pkl"))

    print(f"Artifacts saved → {artifacts_dir}")
    for f in ("scaler.pkl", "pca.pkl", "model.pkl", "label_names.pkl"):
        print(f"  ✓ {f}")


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_training(
    model_type   = "linearsvc",
    n_components = PCA_COMPONENTS,
    run_cv       = True,
    use_hog      = True,
    use_lbp      = True,
    use_hsv      = True,
    artifacts_dir = ARTIFACTS_PATH,
):
    """
    Full pipeline:
        Load → Split → Scale → PCA → (CV) → Train → Evaluate → Save
    """
    X, y, label_names = load_data(use_hog=use_hog, use_lbp=use_lbp, use_hsv=use_hsv)
    if X is None:
        return

    X_train, X_test, y_train, y_test = split_data(X, y)
    del X  # free original array immediately

    X_train, X_test, scaler, pca = preprocess(X_train, X_test, n_components=n_components)

    if run_cv:
        cross_validate(model_type, X_train, y_train)

    model = train_model(X_train, y_train, model_type=model_type)
    evaluate_model(model, X_test, y_test)
    save_artifacts(scaler, pca, model, label_names, artifacts_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Configuration ────────────────────────────────────────────────────────
    MODEL_TYPES   = ["linearsvc", "logreg", "lgbm"]
    N_COMPONENTS  = 500           # PCA: try 200 / 300 / 500
    RUN_CV        = True          # StratifiedKFold cross-validation (3-fold)

    # Feature subset flags — disable any to experiment with subsets
    USE_HOG = True
    USE_LBP = True
    USE_HSV = True
    # ─────────────────────────────────────────────────────────────────────────

    for model_type in MODEL_TYPES:
        print(f"\n{'='*60}\nTraining model: {model_type}\n{'='*60}")
        # Save artifacts in a subfolder for each model
        model_artifacts_dir = os.path.join(ARTIFACTS_PATH, model_type)
        run_training(
            model_type    = model_type,
            n_components  = N_COMPONENTS,
            run_cv        = RUN_CV,
            use_hog       = USE_HOG,
            use_lbp       = USE_LBP,
            use_hsv       = USE_HSV,
            artifacts_dir = model_artifacts_dir,
        )

