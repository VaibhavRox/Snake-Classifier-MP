#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for Snake Classifier Streamlit App
-----------------------------------------------
This script verifies that all components are working correctly.
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_imports():
    """Test if all required modules can be imported."""
    print("=" * 60)
    print("Testing Imports...")
    print("=" * 60)

    try:
        import streamlit
        print("[OK] Streamlit imported successfully")
    except ImportError as e:
        print(f"[FAIL] Streamlit import failed: {e}")
        return False

    try:
        import torch
        import torchvision
        print(f"[OK] PyTorch imported successfully (version: {torch.__version__})")
        print(f"   CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"[FAIL] PyTorch import failed: {e}")
        return False

    try:
        from PIL import Image
        print("[OK] PIL/Pillow imported successfully")
    except ImportError as e:
        print(f"[FAIL] PIL import failed: {e}")
        return False

    try:
        import numpy as np
        import cv2
        from sklearn import __version__ as sklearn_version
        print(f"[OK] NumPy, OpenCV, Scikit-learn imported successfully")
        print(f"   Scikit-learn version: {sklearn_version}")
    except ImportError as e:
        print(f"[FAIL] Scientific libraries import failed: {e}")
        return False

    return True


def test_project_structure():
    """Test if required project files and folders exist."""
    print("\n" + "=" * 60)
    print("Testing Project Structure...")
    print("=" * 60)

    required_files = [
        "app/streamlit_app.py",
        "src/inference/predictor.py",
        "src/features/extractors.py",
        "src/utils/config.py",
        "src/utils/safety.py",
    ]

    required_dirs = [
        "src",
        "src/models",
        "src/models/artifacts",
        "data/snake_reference_images",
    ]

    all_ok = True

    # Check directories
    for dir_path in required_dirs:
        full_path = PROJECT_ROOT / dir_path
        if full_path.exists():
            print(f"[OK] Directory exists: {dir_path}")
        else:
            print(f"[FAIL] Directory missing: {dir_path}")
            all_ok = False

    # Check files
    for file_path in required_files:
        full_path = PROJECT_ROOT / file_path
        if full_path.exists():
            print(f"[OK] File exists: {file_path}")
        else:
            print(f"[FAIL] File missing: {file_path}")
            all_ok = False

    return all_ok


def test_model_artifacts():
    """Test if model artifacts are present."""
    print("\n" + "=" * 60)
    print("Testing Model Artifacts...")
    print("=" * 60)

    model_types = ["logreg", "linearsvc"]
    required_files = ["model.pkl", "scaler.pkl", "label_names.pkl"]

    all_ok = True

    for model_type in model_types:
        print(f"\nChecking {model_type.upper()} artifacts:")
        artifact_dir = PROJECT_ROOT / "src" / "models" / "artifacts" / model_type

        if not artifact_dir.exists():
            print(f"  [FAIL] Directory missing: {artifact_dir}")
            all_ok = False
            continue

        for file_name in required_files:
            file_path = artifact_dir / file_name
            if file_path.exists():
                size_kb = file_path.stat().st_size / 1024
                print(f"  [OK] {file_name} ({size_kb:.2f} KB)")
            else:
                print(f"  [FAIL] {file_name} missing")
                all_ok = False

    return all_ok


def test_reference_images():
    """Test if reference images are available."""
    print("\n" + "=" * 60)
    print("Testing Reference Images...")
    print("=" * 60)

    ref_dir = PROJECT_ROOT / "data" / "snake_reference_images"

    if not ref_dir.exists():
        print(f"[FAIL] Reference directory missing: {ref_dir}")
        return False

    species_dirs = [d for d in ref_dir.iterdir() if d.is_dir()]

    if not species_dirs:
        print("[FAIL] No species directories found")
        return False

    print(f"[OK] Found {len(species_dirs)} species directories:")

    for species_dir in species_dirs[:5]:  # Show first 5
        image_files = list(species_dir.glob("*.jpg")) + \
                      list(species_dir.glob("*.jpeg")) + \
                      list(species_dir.glob("*.png"))
        print(f"  - {species_dir.name}: {len(image_files)} images")

    if len(species_dirs) > 5:
        print(f"  ... and {len(species_dirs) - 5} more")

    return True


def test_classifier_loading():
    """Test if the classifier can be loaded."""
    print("\n" + "=" * 60)
    print("Testing Classifier Loading...")
    print("=" * 60)

    try:
        from src.inference.predictor import SnakeClassifier

        # Try loading LogReg model
        print("Loading LogReg classifier...")
        classifier = SnakeClassifier(model_type="logreg")

        if classifier.model is None:
            print("[FAIL] Classifier model is None")
            return False

        print(f"[OK] Classifier loaded successfully")
        print(f"   Number of classes: {len(classifier.label_names)}")
        print(f"   Sample classes: {classifier.label_names[:3]}")

        return True

    except Exception as e:
        print(f"[FAIL] Classifier loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_extraction():
    """Test if feature extraction works."""
    print("\n" + "=" * 60)
    print("Testing Feature Extraction...")
    print("=" * 60)

    try:
        from src.features.extractors import extract_hog_features
        import numpy as np

        # Find a test image
        ref_dir = PROJECT_ROOT / "data" / "snake_reference_images"
        test_image = None

        for species_dir in ref_dir.iterdir():
            if species_dir.is_dir():
                images = list(species_dir.glob("*.jpg")) + \
                         list(species_dir.glob("*.jpeg")) + \
                         list(species_dir.glob("*.png"))
                if images:
                    test_image = images[0]
                    break

        if test_image is None:
            print("[WARN] No test images found, skipping feature extraction test")
            return True

        print(f"Using test image: {test_image.name}")
        features = extract_hog_features(str(test_image))

        if features is None:
            print("[FAIL] Feature extraction returned None")
            return False

        print(f"[OK] Feature extraction successful")
        print(f"   Feature shape: {features.shape}")
        print(f"   Feature type: {features.dtype}")
        print(f"   Feature range: [{features.min():.3f}, {features.max():.3f}]")

        return True

    except Exception as e:
        print(f"[FAIL] Feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n")
    print("=" * 60)
    print("Snake Classifier - System Tests")
    print("=" * 60)
    print()

    tests = [
        ("Imports", test_imports),
        ("Project Structure", test_project_structure),
        ("Model Artifacts", test_model_artifacts),
        ("Reference Images", test_reference_images),
        ("Classifier Loading", test_classifier_loading),
        ("Feature Extraction", test_feature_extraction),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n[FAIL] Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, result in results:
        status = "[OK] PASS" if result else "[FAIL] FAIL"
        print(f"{status}: {test_name}")

    total_tests = len(results)
    passed_tests = sum(1 for _, result in results if result)

    print("\n" + "=" * 60)
    print(f"Results: {passed_tests}/{total_tests} tests passed")
    print("=" * 60)

    if passed_tests == total_tests:
        print("\n*** All tests passed! You're ready to run the Streamlit app! ***")
        print("\nRun: streamlit run app/streamlit_app.py")
        return 0
    else:
        print("\n*** Some tests failed. Please fix the issues before running the app. ***")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
