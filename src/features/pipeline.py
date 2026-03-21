"""
Feature Extraction Pipeline for Snake Classification.

Supports multiple feature extractors:
- EfficientNet-B0 (1280-dim) - Default
- ResNet50 (2048-dim)

Augmentation is handled separately AFTER train-test split to avoid data leakage.
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
from src.utils.config import (
    DATASET_PATH, PROCESSED_DATA_PATH,
    EFFICIENTNET_FEATURE_DIM, RESNET_FEATURE_DIM,
    AVAILABLE_EXTRACTORS
)
from src.features.extractors import (
    extract_features_batch,
    extract_features_from_array,
    get_feature_dim
)
from src.features.augmentation import generate_augmented_images


def process_dataset(
    extractor="efficientnet",
    max_images_per_class=None,
    max_classes=None,
    use_batch=True,
    batch_size=32
):
    """
    Extract features from ALL original images in the dataset.

    NOTE: Augmentation should be applied AFTER train-test split using
    augment_training_data() to avoid data leakage.

    Parameters
    ----------
    extractor : str
        Feature extractor to use: "efficientnet" or "resnet50"
    max_images_per_class : int or None
        Limit images per class (None = all).
    max_classes : int or None
        Limit number of species folders to process (None = all).
    use_batch : bool
        Whether to use batch processing for speed.
    batch_size : int
        Batch size for feature extraction.

    Returns
    -------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Labels
    label_names : list
        Class names
    image_paths : np.ndarray
        Paths to images
    """
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

    feature_dim = get_feature_dim(extractor)
    print(f"\n{'='*60}")
    print(f"Feature Extraction: {extractor.upper()} ({feature_dim}-dim)")
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

    print(f"Total images to process: {len(all_paths)}")

    # Extract features
    if use_batch:
        print(f"Extracting features in batches (batch_size={batch_size})...")
        features, failed = extract_features_batch(
            all_paths, extractor=extractor, batch_size=batch_size
        )

        # Remove failed indices from labels
        if failed:
            print(f"Warning: {len(failed)} images failed to process")
            valid_mask = np.ones(len(all_labels), dtype=bool)
            for idx in failed:
                valid_mask[idx] = False
            all_labels = [l for i, l in enumerate(all_labels) if valid_mask[i]]
            all_paths = [p for i, p in enumerate(all_paths) if valid_mask[i]]

        X = features
        y = np.array(all_labels)
        image_paths = np.array(all_paths)
    else:
        # Sequential processing (slower but more memory efficient)
        from src.features.extractors import extract_features
        features = []
        valid_paths = []
        valid_labels = []

        for img_path, label in tqdm(zip(all_paths, all_labels), total=len(all_paths), desc="Extracting"):
            feat = extract_features(img_path, extractor=extractor)
            if feat is not None:
                features.append(feat)
                valid_paths.append(img_path)
                valid_labels.append(label)

        X = np.array(features, dtype=np.float32)
        y = np.array(valid_labels)
        image_paths = np.array(valid_paths)

    print(f"\nExtraction complete. Feature matrix: {X.shape} | dtype: {X.dtype}")
    print(f"Labels: {y.shape}")

    # Print class distribution
    print("\n" + "="*60)
    print("DATASET DISTRIBUTION")
    print("="*60)
    for species, count in sorted(class_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {species:<40} : {count:>4} images")
    if len(class_counts) > 10:
        print(f"  ... and {len(class_counts) - 10} more classes")
    print("="*60)
    print(f"  Total: {sum(class_counts.values())} images across {len(class_counts)} classes")
    print("="*60 + "\n")

    # Save with extractor name in filename
    save_path = os.path.join(PROCESSED_DATA_PATH, f"features_{extractor}.npz")
    np.savez_compressed(
        save_path,
        features=X,
        labels=y,
        label_names=np.array(label_names),
        image_paths=image_paths,
        extractor=extractor
    )
    print(f"Saved -> {save_path}")

    return X, y, label_names, image_paths


def process_dataset_all_extractors(
    max_images_per_class=None,
    max_classes=None,
    batch_size=32
):
    """
    Extract features using ALL available extractors for comparison.

    Parameters
    ----------
    max_images_per_class : int or None
        Limit images per class (None = all).
    max_classes : int or None
        Limit number of species folders to process (None = all).
    batch_size : int
        Batch size for feature extraction.

    Returns
    -------
    dict
        Dictionary mapping extractor name to (X, y, label_names, image_paths)
    """
    results = {}

    for extractor in AVAILABLE_EXTRACTORS:
        print(f"\n{'#'*60}")
        print(f"# Processing with {extractor.upper()}")
        print(f"{'#'*60}")

        X, y, label_names, image_paths = process_dataset(
            extractor=extractor,
            max_images_per_class=max_images_per_class,
            max_classes=max_classes,
            use_batch=True,
            batch_size=batch_size
        )

        results[extractor] = {
            'X': X,
            'y': y,
            'label_names': label_names,
            'image_paths': image_paths
        }

    return results


def load_features(extractor="efficientnet"):
    """
    Load preprocessed features for a given extractor.

    Parameters
    ----------
    extractor : str
        Feature extractor name.

    Returns
    -------
    X, y, label_names, image_paths or None if not found.
    """
    data_file = os.path.join(PROCESSED_DATA_PATH, f"features_{extractor}.npz")

    if not os.path.exists(data_file):
        # Try old filename format for backward compatibility
        old_file = os.path.join(PROCESSED_DATA_PATH, "features.npz")
        if os.path.exists(old_file):
            data_file = old_file
        else:
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


def augment_training_data(
    X_train, y_train, image_paths_train, label_names,
    extractor="efficientnet",
    augment_factor=2,
    max_aug_ratio=0.5,
    verbose=True
):
    """
    Apply augmentation to training data only (AFTER train-test split).

    Parameters
    ----------
    X_train : np.ndarray
        Original training features
    y_train : np.ndarray
        Original training labels
    image_paths_train : np.ndarray
        Paths to original training images
    label_names : np.ndarray
        Class names
    extractor : str
        Feature extractor to use for augmented images
    augment_factor : int
        Number of augmented copies per image
    max_aug_ratio : float
        Maximum ratio of augmented to real images (0.5 = 50%)

    Returns
    -------
    X_train_aug : np.ndarray
        Combined original + augmented features
    y_train_aug : np.ndarray
        Combined labels
    stats : dict
        Per-class augmentation statistics
    """
    unique_labels = np.unique(y_train)
    class_counts = {int(label): np.sum(y_train == label) for label in unique_labels}

    aug_features = []
    aug_labels = []
    stats = {}

    if verbose:
        print("\n" + "="*70)
        print(f"AUGMENTATION (Training Set Only) - {extractor.upper()}")
        print("="*70)
        print(f"{'Class':<40} | {'Real':>6} | {'Aug':>6} | {'Total':>6} | {'Ratio':>6}")
        print("-"*70)

    for label in tqdm(unique_labels, desc="Augmenting", disable=not verbose):
        label = int(label)
        real_count = class_counts[label]
        class_name = label_names[label] if label < len(label_names) else f"Class_{label}"

        class_mask = y_train == label
        class_paths = image_paths_train[class_mask]

        max_aug = int(real_count * max_aug_ratio)
        actual_aug_factor = min(augment_factor, max(1, max_aug // real_count))

        aug_count = 0

        if actual_aug_factor > 0:
            for img_path in class_paths:
                img_bgr = cv2.imread(img_path)
                if img_bgr is None:
                    continue

                aug_images = generate_augmented_images(img_bgr, num_augments=actual_aug_factor)

                for aug_img in aug_images:
                    feat = extract_features_from_array(aug_img, extractor=extractor)
                    if feat is not None:
                        aug_features.append(feat)
                        aug_labels.append(label)
                        aug_count += 1

                        if aug_count >= max_aug:
                            break

                if aug_count >= max_aug:
                    break

        total = real_count + aug_count
        ratio = aug_count / real_count if real_count > 0 else 0
        stats[class_name] = {
            'real': real_count,
            'augmented': aug_count,
            'total': total,
            'ratio': ratio
        }

        if verbose:
            print(f"  {class_name:<38} | {real_count:>6} | {aug_count:>6} | {total:>6} | {ratio:>5.2f}")

    if verbose:
        print("-"*70)
        total_real = sum(s['real'] for s in stats.values())
        total_aug = sum(s['augmented'] for s in stats.values())
        print(f"  {'TOTAL':<38} | {total_real:>6} | {total_aug:>6} | {total_real+total_aug:>6} | {total_aug/total_real:.2f}")
        print("="*70 + "\n")

    # Combine original + augmented
    if aug_features:
        X_aug = np.array(aug_features, dtype=np.float32)
        y_aug = np.array(aug_labels)
        X_combined = np.vstack([X_train, X_aug])
        y_combined = np.concatenate([y_train, y_aug])
    else:
        X_combined = X_train
        y_combined = y_train

    return X_combined, y_combined, stats


if __name__ == "__main__":
    # Extract features using all extractors for comparison
    process_dataset_all_extractors(max_images_per_class=None, max_classes=None)
