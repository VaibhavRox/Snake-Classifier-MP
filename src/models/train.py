"""
Snake Classifier — Training Pipeline (HOG+LBP+HSV Features)
================================================================

Pipeline: Load -> Split -> Scale -> Tune -> Train -> Evaluate -> Save

Models: LinearSVC, Logistic Regression
"""

import os
import time
import numpy as np
import joblib
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, train_test_split, GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, top_k_accuracy_score

from src.utils.config import (
    PROCESSED_DATA_PATH, ARTIFACTS_PATH, RANDOM_SEED, TEST_SPLIT,
    FEATURE_DIM, MODEL_TYPES, PARAM_GRIDS, CV_FOLDS, TUNING_FOLDS
)
from src.features.pipeline import process_dataset, load_features


def load_data():
    """Load features, running extraction if needed."""
    print("Loading HOG+LBP+HSV features...")

    X, y, label_names, image_paths = load_features()

    if X is None:
        print("Features not found — extracting...")
        X, y, label_names, image_paths = process_dataset()

    print(f"Loaded: {X.shape} | Classes: {len(np.unique(y))}")
    return X, y, label_names, image_paths


def split_data(X, y, image_paths=None):
    """Stratified train/test split."""
    print(f"Splitting: {int((1-TEST_SPLIT)*100)}/{int(TEST_SPLIT*100)}...")

    if image_paths is not None:
        X_train, X_test, y_train, y_test, _, _ = train_test_split(
            X, y, image_paths, test_size=TEST_SPLIT, stratify=y, random_state=RANDOM_SEED
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SPLIT, stratify=y, random_state=RANDOM_SEED
        )

    print(f"  Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def preprocess(X_train, X_test):
    """Apply StandardScaler."""
    print("Scaling features...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, scaler


def tune_hyperparameters(X_train, y_train, model_type):
    """GridSearchCV hyperparameter tuning."""
    print(f"\nTuning {model_type.upper()} ({TUNING_FOLDS}-fold CV)...")

    skf = StratifiedKFold(n_splits=TUNING_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    if model_type == "linearsvc":
        base_model = LinearSVC(max_iter=10000, random_state=RANDOM_SEED, class_weight="balanced", dual=True)
    elif model_type == "logreg":
        base_model = LogisticRegression(class_weight="balanced", random_state=RANDOM_SEED)
    else:
        return None, None

    param_grid = PARAM_GRIDS.get(model_type, {})

    t0 = time.time()
    grid = GridSearchCV(base_model, param_grid, cv=skf, scoring='accuracy', n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)

    print(f"  Best params: {grid.best_params_}")
    print(f"  Best CV: {grid.best_score_*100:.2f}%")
    print(f"  Time: {time.time()-t0:.1f}s")

    return grid.best_estimator_, grid.best_params_


def train_model(X_train, y_train, model_type, best_params=None):
    """Train model with best params."""
    params = best_params or {}

    if model_type == "linearsvc":
        model = LinearSVC(
            C=params.get('C', 1.0),
            loss=params.get('loss', 'squared_hinge'),
            max_iter=10000, class_weight="balanced", dual=True, random_state=RANDOM_SEED
        )
    elif model_type == "logreg":
        model = LogisticRegression(
            C=params.get('C', 1.0),
            solver=params.get('solver', 'lbfgs'),
            penalty=params.get('penalty', 'l2'),
            max_iter=params.get('max_iter', 2000),
            class_weight="balanced", random_state=RANDOM_SEED
        )
    else:
        return None, 0

    t0 = time.time()
    model.fit(X_train, y_train)
    return model, time.time() - t0


def evaluate_model(model, X_test, y_test):
    """Evaluate Top-1, Top-3, Top-5 accuracy."""
    y_pred = model.predict(X_test)
    top1 = accuracy_score(y_test, y_pred)

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
    elif hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_test)
    else:
        return top1, None, None

    n_classes = scores.shape[1] if len(scores.shape) > 1 else len(np.unique(y_test))
    top3 = top_k_accuracy_score(y_test, scores, k=min(3, n_classes))
    top5 = top_k_accuracy_score(y_test, scores, k=min(5, n_classes))

    return top1, top3, top5


def cross_validate(model_type, X_train, y_train, best_params=None):
    """Final cross-validation with best params."""
    params = best_params or {}

    if model_type == "linearsvc":
        cv_model = LinearSVC(
            C=params.get('C', 1.0), loss=params.get('loss', 'squared_hinge'),
            max_iter=10000, class_weight="balanced", dual=True, random_state=RANDOM_SEED
        )
    elif model_type == "logreg":
        cv_model = LogisticRegression(
            C=params.get('C', 1.0), solver=params.get('solver', 'lbfgs'),
            penalty=params.get('penalty', 'l2'), max_iter=params.get('max_iter', 2000),
            class_weight="balanced", random_state=RANDOM_SEED
        )
    else:
        return None, None

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_val_score(cv_model, X_train, y_train, cv=skf, scoring="accuracy", n_jobs=-1)
    return scores.mean(), scores.std()


def save_artifacts(scaler, model, label_names, best_params, model_type):
    """Save model artifacts."""
    artifacts_dir = os.path.join(ARTIFACTS_PATH, model_type)
    os.makedirs(artifacts_dir, exist_ok=True)

    joblib.dump(scaler, os.path.join(artifacts_dir, "scaler.pkl"))
    joblib.dump(model, os.path.join(artifacts_dir, "model.pkl"))
    joblib.dump(label_names, os.path.join(artifacts_dir, "label_names.pkl"))
    if best_params:
        joblib.dump(best_params, os.path.join(artifacts_dir, "best_params.pkl"))

    print(f"  Saved -> {artifacts_dir}")


def run_training(model_type="linearsvc", run_tuning=True, run_cv=True):
    """Full training pipeline for a single model."""
    # Load
    X, y, label_names, image_paths = load_data()
    if X is None:
        return None

    # Split
    X_train, X_test, y_train, y_test = split_data(X, y, image_paths)

    # Scale
    X_train, X_test, scaler = preprocess(X_train, X_test)

    # Tune
    best_params = None
    if run_tuning:
        _, best_params = tune_hyperparameters(X_train, y_train, model_type)

    # Train
    model, train_time = train_model(X_train, y_train, model_type, best_params)
    if model is None:
        return None

    # Evaluate
    top1, top3, top5 = evaluate_model(model, X_test, y_test)

    print(f"\n{'='*50}")
    print(f"  {model_type.upper()} Results")
    print(f"{'='*50}")
    print(f"  Top-1: {top1*100:.2f}%")
    print(f"  Top-3: {top3*100:.2f}%")
    print(f"  Top-5: {top5*100:.2f}%")

    # CV
    if run_cv:
        cv_mean, cv_std = cross_validate(model_type, X_train, y_train, best_params)
        if cv_mean:
            print(f"  CV: {cv_mean*100:.2f}% +/- {cv_std*100:.2f}%")

    print(f"  Train time: {train_time:.1f}s")
    print(f"{'='*50}")

    # Save
    save_artifacts(scaler, model, label_names, best_params, model_type)

    return {'top1': top1, 'top3': top3, 'top5': top5, 'model': model}


if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# SNAKE CLASSIFIER - HOG+LBP+HSV Features")
    print("#"*60)

    results = {}
    for model_type in MODEL_TYPES:
        print(f"\n{'='*60}")
        print(f"Training: {model_type.upper()}")
        print("="*60)
        result = run_training(model_type, run_tuning=True, run_cv=True)
        if result:
            results[model_type] = result

    # Summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    for name, res in results.items():
        print(f"  {name.upper()}: Top-1={res['top1']*100:.2f}%  Top-3={res['top3']*100:.2f}%")

    best = max(results.items(), key=lambda x: x[1]['top1'])
    print(f"\nBest: {best[0].upper()} ({best[1]['top1']*100:.2f}%)")
