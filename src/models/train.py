"""
Snake Classifier — Training Pipeline with Ablation Study
=========================================================

Supports comparison of multiple feature extractors:
- EfficientNet-B0 (1280-dim)
- ResNet50 (2048-dim)

With multiple classifiers:
- LinearSVC
- Logistic Regression
- LightGBM

Pipeline: Load -> Split -> Scale -> Tune -> Train -> Evaluate -> Compare
"""

import os
import time
import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, train_test_split, GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, top_k_accuracy_score

from src.utils.config import (
    PROCESSED_DATA_PATH, ARTIFACTS_PATH, RANDOM_SEED, TEST_SPLIT,
    AVAILABLE_EXTRACTORS, MODEL_TYPES, PARAM_GRIDS, CV_FOLDS, TUNING_FOLDS
)
from src.features.pipeline import process_dataset, load_features
from src.features.extractors import get_feature_dim


# =============================================================================
# Data Loading
# =============================================================================

def load_data(extractor="efficientnet"):
    """
    Load preprocessed features for a given extractor.
    If features don't exist, runs extraction pipeline first.

    Returns (X, y, label_names, image_paths) or (None, None, None, None) on failure.
    """
    print(f"\nLoading {extractor.upper()} features...")

    X, y, label_names, image_paths = load_features(extractor)

    if X is None:
        print(f"Features not found — running extraction pipeline for {extractor}...")
        X, y, label_names, image_paths = process_dataset(
            extractor=extractor,
            max_images_per_class=None,
            max_classes=None
        )

    if X is not None:
        print(f"Loaded: {X.shape} samples | dtype: {X.dtype} | classes: {len(np.unique(y))}")

    return X, y, label_names, image_paths


# =============================================================================
# Data Splitting
# =============================================================================

def split_data(X, y, image_paths=None):
    """
    Stratified train/test split BEFORE any preprocessing to prevent data leakage.
    """
    print(f"Splitting data — stratified {int((1-TEST_SPLIT)*100)}/{int(TEST_SPLIT*100)}...")

    if image_paths is not None:
        X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(
            X, y, image_paths, test_size=TEST_SPLIT, stratify=y, random_state=RANDOM_SEED
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SPLIT, stratify=y, random_state=RANDOM_SEED
        )
        paths_train, paths_test = None, None

    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, paths_train, paths_test


# =============================================================================
# Preprocessing (StandardScaler only - NO PCA)
# =============================================================================

def preprocess(X_train, X_test):
    """Apply StandardScaler fitted on training data only."""
    print("Scaling features (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"  Scaled: Train {X_train_scaled.shape}  |  Test {X_test_scaled.shape}")
    return X_train_scaled, X_test_scaled, scaler


# =============================================================================
# Hyperparameter Tuning with GridSearchCV
# =============================================================================

def tune_hyperparameters(X_train, y_train, model_type="linearsvc", n_splits=TUNING_FOLDS):
    """
    Perform GridSearchCV with StratifiedKFold to find optimal hyperparameters.

    Returns (best_model, best_params, tuning_results).
    """
    model_type = model_type.lower()
    print(f"\nTuning {model_type.upper()} ({n_splits}-fold CV)...")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    if model_type == "linearsvc":
        base_model = LinearSVC(
            max_iter=10000,
            random_state=RANDOM_SEED,
            class_weight="balanced",
            dual=True
        )
        param_grid = PARAM_GRIDS.get('linearsvc', {'C': [1]})

    elif model_type == "logreg":
        base_model = LogisticRegression(
            class_weight="balanced",
            random_state=RANDOM_SEED
        )
        param_grid = PARAM_GRIDS.get('logreg', {'C': [1]})

    elif model_type == "lgbm":
        try:
            import lightgbm as lgb
            base_model = lgb.LGBMClassifier(
                class_weight="balanced",
                random_state=RANDOM_SEED,
                verbose=-1,
                n_jobs=-1
            )
            param_grid = PARAM_GRIDS.get('lgbm', {'n_estimators': [100]})
        except ImportError:
            print("LightGBM not installed — skipping")
            return None, None, None

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    t0 = time.time()
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=skf,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0,
        return_train_score=True
    )
    grid_search.fit(X_train, y_train)
    elapsed = time.time() - t0

    print(f"  Best params: {grid_search.best_params_}")
    print(f"  Best CV score: {grid_search.best_score_*100:.2f}%")
    print(f"  Tuning time: {elapsed:.1f}s")

    return grid_search.best_estimator_, grid_search.best_params_, {
        'best_score': grid_search.best_score_,
        'best_params': grid_search.best_params_,
        'tuning_time': elapsed
    }


# =============================================================================
# Model Training
# =============================================================================

def train_model(X_train, y_train, model_type="linearsvc", best_params=None):
    """
    Train a model with the given (or default) hyperparameters.

    Returns (fitted_model, train_time).
    """
    model_type = model_type.lower()
    params = best_params or {}

    if model_type == "linearsvc":
        model = LinearSVC(
            C=params.get('C', 1.0),
            loss=params.get('loss', 'squared_hinge'),
            max_iter=params.get('max_iter', 10000),
            class_weight="balanced",
            dual=True,
            random_state=RANDOM_SEED
        )

    elif model_type == "logreg":
        model = LogisticRegression(
            C=params.get('C', 1.0),
            solver=params.get('solver', 'lbfgs'),
            penalty=params.get('penalty', 'l2'),
            max_iter=params.get('max_iter', 2000),
            class_weight="balanced",
            random_state=RANDOM_SEED
        )

    elif model_type == "lgbm":
        try:
            import lightgbm as lgb
            model = lgb.LGBMClassifier(
                n_estimators=params.get('n_estimators', 300),
                learning_rate=params.get('learning_rate', 0.05),
                num_leaves=params.get('num_leaves', 31),
                max_depth=params.get('max_depth', -1),
                class_weight="balanced",
                random_state=RANDOM_SEED,
                verbose=-1,
                n_jobs=-1
            )
        except ImportError:
            print("LightGBM not installed")
            return None, 0

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    return model, train_time


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model and return Top-1, Top-3, Top-5 accuracy.
    """
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


def cross_validate_model(model_type, X_train, y_train, best_params=None, n_splits=CV_FOLDS):
    """
    Run StratifiedKFold cross-validation with best hyperparameters.

    Returns (mean_score, std_score, fold_scores).
    """
    params = best_params or {}

    if model_type == "linearsvc":
        cv_model = LinearSVC(
            C=params.get('C', 1.0),
            loss=params.get('loss', 'squared_hinge'),
            max_iter=params.get('max_iter', 10000),
            class_weight="balanced",
            dual=True,
            random_state=RANDOM_SEED
        )
    elif model_type == "logreg":
        cv_model = LogisticRegression(
            C=params.get('C', 1.0),
            solver=params.get('solver', 'lbfgs'),
            penalty=params.get('penalty', 'l2'),
            max_iter=params.get('max_iter', 2000),
            class_weight="balanced",
            random_state=RANDOM_SEED
        )
    elif model_type == "lgbm":
        try:
            import lightgbm as lgb
            cv_model = lgb.LGBMClassifier(
                n_estimators=params.get('n_estimators', 300),
                learning_rate=params.get('learning_rate', 0.05),
                num_leaves=params.get('num_leaves', 31),
                max_depth=params.get('max_depth', -1),
                class_weight="balanced",
                random_state=RANDOM_SEED,
                verbose=-1,
                n_jobs=-1
            )
        except ImportError:
            return None, None, None
    else:
        return None, None, None

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_val_score(cv_model, X_train, y_train, cv=skf, scoring="accuracy", n_jobs=-1)

    return scores.mean(), scores.std(), scores


# =============================================================================
# Save Artifacts
# =============================================================================

def save_artifacts(scaler, model, label_names, best_params, extractor, model_type, base_dir=ARTIFACTS_PATH):
    """Save trained model artifacts."""
    artifacts_dir = os.path.join(base_dir, extractor, model_type)
    os.makedirs(artifacts_dir, exist_ok=True)

    joblib.dump(scaler, os.path.join(artifacts_dir, "scaler.pkl"))
    joblib.dump(model, os.path.join(artifacts_dir, "model.pkl"))
    joblib.dump(label_names, os.path.join(artifacts_dir, "label_names.pkl"))
    if best_params:
        joblib.dump(best_params, os.path.join(artifacts_dir, "best_params.pkl"))
    joblib.dump(extractor, os.path.join(artifacts_dir, "extractor.pkl"))

    print(f"  Saved artifacts -> {artifacts_dir}")


# =============================================================================
# Single Extractor + Model Training
# =============================================================================

def run_training_single(
    extractor="efficientnet",
    model_type="linearsvc",
    run_tuning=True,
    run_cv=True,
    save=True
):
    """
    Train a single model with a single feature extractor.

    Returns dict with results.
    """
    # Load data
    X, y, label_names, image_paths = load_data(extractor)
    if X is None:
        return None

    # Split
    X_train, X_test, y_train, y_test, _, _ = split_data(X, y, image_paths)
    del X

    # Scale
    X_train, X_test, scaler = preprocess(X_train, X_test)

    # Tune hyperparameters
    best_params = None
    tuning_time = 0
    if run_tuning:
        _, best_params, tuning_info = tune_hyperparameters(X_train, y_train, model_type)
        if tuning_info:
            tuning_time = tuning_info['tuning_time']

    # Train
    model, train_time = train_model(X_train, y_train, model_type, best_params)
    if model is None:
        return None

    # Evaluate
    top1, top3, top5 = evaluate_model(model, X_test, y_test)

    # Cross-validation
    cv_mean, cv_std = None, None
    if run_cv:
        cv_mean, cv_std, _ = cross_validate_model(model_type, X_train, y_train, best_params)

    # Save
    if save:
        save_artifacts(scaler, model, label_names, best_params, extractor, model_type)

    result = {
        'extractor': extractor,
        'model': model_type,
        'feature_dim': get_feature_dim(extractor),
        'top1': top1,
        'top3': top3,
        'top5': top5,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'best_params': best_params,
        'train_time': train_time,
        'tuning_time': tuning_time,
        'total_time': train_time + tuning_time
    }

    return result


# =============================================================================
# Full Ablation Study
# =============================================================================

def run_ablation_study(
    extractors=None,
    models=None,
    run_tuning=True,
    run_cv=True,
    save_best_only=True
):
    """
    Run full ablation study comparing all extractors and models.

    Parameters
    ----------
    extractors : list
        Feature extractors to compare. Default: all available.
    models : list
        Model types to compare. Default: all available.
    run_tuning : bool
        Whether to run hyperparameter tuning.
    run_cv : bool
        Whether to run cross-validation.
    save_best_only : bool
        If True, only save artifacts for the best model.

    Returns
    -------
    results_df : pd.DataFrame
        Comparison table with all results.
    best_result : dict
        Best performing configuration.
    """
    extractors = extractors or AVAILABLE_EXTRACTORS
    models = models or MODEL_TYPES

    print("\n" + "="*70)
    print("ABLATION STUDY: Feature Extractors x Classifiers")
    print("="*70)
    print(f"Extractors: {extractors}")
    print(f"Models: {models}")
    print("="*70 + "\n")

    all_results = []

    for extractor in extractors:
        print(f"\n{'#'*70}")
        print(f"# EXTRACTOR: {extractor.upper()} ({get_feature_dim(extractor)}-dim)")
        print(f"{'#'*70}")

        # Load data once per extractor
        X, y, label_names, image_paths = load_data(extractor)
        if X is None:
            print(f"Failed to load data for {extractor}")
            continue

        # Split once per extractor
        X_train, X_test, y_train, y_test, _, _ = split_data(X, y, image_paths)
        del X

        # Scale once per extractor
        X_train_scaled, X_test_scaled, scaler = preprocess(X_train, X_test)

        for model_type in models:
            print(f"\n{'-'*50}")
            print(f"Model: {model_type.upper()}")
            print(f"{'-'*50}")

            # Tune
            best_params = None
            tuning_time = 0
            if run_tuning:
                _, best_params, tuning_info = tune_hyperparameters(
                    X_train_scaled, y_train, model_type
                )
                if tuning_info:
                    tuning_time = tuning_info.get('tuning_time', 0)

            # Train
            model, train_time = train_model(
                X_train_scaled, y_train, model_type, best_params
            )
            if model is None:
                continue

            # Evaluate
            top1, top3, top5 = evaluate_model(model, X_test_scaled, y_test)

            print(f"  Top-1: {top1*100:.2f}%  |  Top-3: {top3*100:.2f}%  |  Top-5: {top5*100:.2f}%")

            # Cross-validation
            cv_mean, cv_std = None, None
            if run_cv:
                cv_mean, cv_std, _ = cross_validate_model(
                    model_type, X_train_scaled, y_train, best_params
                )
                if cv_mean is not None:
                    print(f"  CV: {cv_mean*100:.2f}% +/- {cv_std*100:.2f}%")

            result = {
                'Extractor': extractor.upper(),
                'Model': model_type.upper(),
                'Feature Dim': get_feature_dim(extractor),
                'Top-1 (%)': round(top1 * 100, 2),
                'Top-3 (%)': round(top3 * 100, 2) if top3 else None,
                'Top-5 (%)': round(top5 * 100, 2) if top5 else None,
                'CV Mean (%)': round(cv_mean * 100, 2) if cv_mean else None,
                'CV Std (%)': round(cv_std * 100, 2) if cv_std else None,
                'Train Time (s)': round(train_time, 1),
                'Tune Time (s)': round(tuning_time, 1),
                'Best Params': str(best_params) if best_params else 'default',
                '_model': model,
                '_scaler': scaler,
                '_label_names': label_names,
                '_best_params': best_params,
                '_top1': top1
            }
            all_results.append(result)

    if not all_results:
        print("No results to show!")
        return None, None

    # Create comparison DataFrame
    display_cols = [
        'Extractor', 'Model', 'Feature Dim', 'Top-1 (%)', 'Top-3 (%)',
        'Top-5 (%)', 'CV Mean (%)', 'CV Std (%)', 'Train Time (s)'
    ]
    results_df = pd.DataFrame(all_results)[display_cols]

    # Find best result
    best_result = max(all_results, key=lambda x: x['_top1'])

    # Print comparison table
    print("\n" + "="*100)
    print("ABLATION STUDY RESULTS")
    print("="*100)
    print(results_df.to_string(index=False))
    print("="*100)

    # Print best model
    print(f"\nBEST MODEL:")
    print(f"  Extractor: {best_result['Extractor']}")
    print(f"  Model: {best_result['Model']}")
    print(f"  Top-1 Accuracy: {best_result['Top-1 (%)']}%")
    print(f"  Best Params: {best_result['Best Params']}")

    # Save artifacts
    if save_best_only:
        # Save only the best model
        save_artifacts(
            best_result['_scaler'],
            best_result['_model'],
            best_result['_label_names'],
            best_result['_best_params'],
            best_result['Extractor'].lower(),
            best_result['Model'].lower()
        )
    else:
        # Save all models
        for result in all_results:
            save_artifacts(
                result['_scaler'],
                result['_model'],
                result['_label_names'],
                result['_best_params'],
                result['Extractor'].lower(),
                result['Model'].lower()
            )

    return results_df, best_result


# =============================================================================
# Orchestrator for Single Training Run
# =============================================================================

def run_training(
    extractor="efficientnet",
    model_type="linearsvc",
    run_tuning=True,
    run_cv=True,
    artifacts_dir=ARTIFACTS_PATH,
):
    """
    Full training pipeline for a single extractor + model combination.
    """
    result = run_training_single(
        extractor=extractor,
        model_type=model_type,
        run_tuning=run_tuning,
        run_cv=run_cv,
        save=True
    )

    if result is None:
        return None

    # Print summary
    print(f"\n{'='*60}")
    print(f"TRAINING SUMMARY: {result['extractor'].upper()} + {result['model'].upper()}")
    print(f"{'='*60}")
    print(f"  Feature dim: {result['feature_dim']}")
    print(f"  Best params: {result['best_params']}")
    print(f"  Top-1: {result['top1']*100:.2f}%")
    print(f"  Top-3: {result['top3']*100:.2f}%")
    print(f"  Top-5: {result['top5']*100:.2f}%")
    if result['cv_mean']:
        print(f"  CV: {result['cv_mean']*100:.2f}% +/- {result['cv_std']*100:.2f}%")
    print(f"  Total time: {result['total_time']:.1f}s")
    print(f"{'='*60}\n")

    return result


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    # Run full ablation study
    print("\n" + "#"*70)
    print("# SNAKE CLASSIFIER - ABLATION STUDY")
    print("# Comparing: EfficientNet-B0 vs ResNet50")
    print("# Models: LinearSVC, Logistic Regression, LightGBM")
    print("#"*70)

    results_df, best = run_ablation_study(
        extractors=AVAILABLE_EXTRACTORS,  # ["resnet50", "efficientnet"]
        models=MODEL_TYPES,               # ["linearsvc", "logreg", "lgbm"]
        run_tuning=True,
        run_cv=True,
        save_best_only=False  # Save all models for comparison
    )

    if results_df is not None:
        # Save results to CSV
        results_path = os.path.join(ARTIFACTS_PATH, "ablation_results.csv")
        results_df.to_csv(results_path, index=False)
        print(f"\nResults saved to: {results_path}")
