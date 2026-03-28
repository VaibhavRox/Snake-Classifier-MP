"""
Feature Extraction Pipeline using HOG, LBP, and HSV descriptors.

Extracts 1280-dimensional features from images using combined texture, gradient, and color analysis.
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
from src.utils.config import DATASET_PATH, PROCESSED_DATA_PATH, FEATURE_DIM
from src.features.extractors import extract_hog_features, extract_lbp_features, extract_hsv_features_batch
from src.features.augmentation import generate_augmented_images


def process_dataset(max_images_per_class=None, max_classes=None, use_batch=True, batch_size=32):
    """
    Extract HOG+LBP+HSV features from all images in the dataset.

    Parameters
    ----------
    max_images_per_class : int or None
        Limit images per class (None = all).
    max_classes : int or None
        Limit number of species folders (None = all).
    use_batch : bool
        Use batch processing for speed.
    batch_size : int
        Batch size for extraction.

    Returns
    -------
    X, y, label_names, image_paths
    """
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Feature Extraction: HOG+LBP+HSV ({FEATURE_DIM}-dim)")
    print(f"{'='*60}")

    # Collect species folders
    species_folders = sorted([
        d for d in os.listdir(DATASET_PATH)
        if os.path.isdir(os.path.join(DATASET_PATH, d))
    ])

    if max_classes:
        species_folders = species_folders[:max_classes]

    print(f"Found {len(species_folders)} species")

    # Collect all image paths and labels
    all_paths = []
    all_labels = []
    label_names = []
    class_counts = {}

    for label_idx, species in enumerate(species_folders):
        species_dir = os.path.join(DATASET_PATH, species)
        label_names.append(species)

        image_files = [
            f for f in os.listdir(species_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        if max_images_per_class:
            image_files = image_files[:max_images_per_class]

        class_counts[species] = len(image_files)

        for img_file in image_files:
            img_path = os.path.join(species_dir, img_file)
            all_paths.append(img_path)
            all_labels.append(label_idx)

    print(f"Total images: {len(all_paths)}")

    # Extract features
    if use_batch:
        print(f"Extracting HOG features in batches (batch_size={batch_size})...")
        features, failed = extract_hsv_features_batch(all_paths, batch_size=batch_size)

        if failed:
            print(f"Warning: {len(failed)} images failed")
            valid_mask = np.ones(len(all_labels), dtype=bool)
            for idx in failed:
                valid_mask[idx] = False
            all_labels = [l for i, l in enumerate(all_labels) if valid_mask[i]]
            all_paths = [p for i, p in enumerate(all_paths) if valid_mask[i]]

        X = features
        y = np.array(all_labels)
        image_paths = np.array(all_paths)
    else:
        features = []
        valid_paths = []
        valid_labels = []

        for img_path, label in tqdm(zip(all_paths, all_labels), total=len(all_paths)):
            feat = extract_hog_features(img_path)
            if feat is not None:
                features.append(feat)
                valid_paths.append(img_path)
                valid_labels.append(label)

        X = np.array(features, dtype=np.float32)
        y = np.array(valid_labels)
        image_paths = np.array(valid_paths)

    print(f"\nExtraction complete: {X.shape}")

    # Save
    save_path = os.path.join(PROCESSED_DATA_PATH, "features.npz")
    np.savez_compressed(
        save_path,
        features=X,
        labels=y,
        label_names=np.array(label_names),
        image_paths=image_paths
    )
    print(f"Saved -> {save_path}")

    return X, y, label_names, image_paths


def load_features():
    """Load preprocessed features."""
    data_file = os.path.join(PROCESSED_DATA_PATH, "features.npz")

    if not os.path.exists(data_file):
        return None, None, None, None

    try:
        data = np.load(data_file, allow_pickle=True)
        X = data["features"].astype(np.float32)
        y = data["labels"]
        label_names = data["label_names"]
        image_paths = data.get("image_paths", None)
        return X, y, label_names, image_paths
    except Exception as e:
        print(f"Error loading features: {e}")
        return None, None, None, None


if __name__ == "__main__":
    process_dataset()
