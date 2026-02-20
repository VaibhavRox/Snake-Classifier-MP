
import os
import numpy as np
from tqdm import tqdm
from src.utils.config import DATASET_PATH, PROCESSED_DATA_PATH
from src.features.extractors import extract_selected_features


def process_dataset(
    max_images_per_class=None,
    max_classes=None,
    use_hog=True,
    use_lbp=True,
    use_hsv=True,
):
    """
    Iterates through the dataset, extracts features, and saves them.

    Parameters
    ----------
    max_images_per_class : int or None
        Limit images per class (None = all).
    max_classes : int or None
        Limit number of species folders to process (None = all).
    use_hog / use_lbp / use_hsv : bool
        Feature subset flags — any combination is valid.
    """
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

    features    = []
    labels      = []
    label_names = []

    # Collect species folders
    species_folders = sorted([
        d for d in os.listdir(DATASET_PATH)
        if os.path.isdir(os.path.join(DATASET_PATH, d))
    ])

    if max_classes:
        species_folders = species_folders[:max_classes]

    active = [n for n, f in [("HOG", use_hog), ("LBP", use_lbp), ("HSV", use_hsv)] if f]
    print(f"Found {len(species_folders)} species. Active features: {', '.join(active)}")

    for label_idx, species in enumerate(tqdm(species_folders, desc="Processing Species")):
        species_dir = os.path.join(DATASET_PATH, species)
        label_names.append(species)

        image_files = [
            f for f in os.listdir(species_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        if max_images_per_class:
            image_files = image_files[:max_images_per_class]

        for img_file in image_files:
            img_path = os.path.join(species_dir, img_file)
            feat_vector = extract_selected_features(
                img_path, use_hog=use_hog, use_lbp=use_lbp, use_hsv=use_hsv
            )
            if feat_vector is not None:
                features.append(feat_vector)
                labels.append(label_idx)

    # Convert — store as float32 to halve memory vs float64
    X = np.array(features, dtype=np.float32)
    y = np.array(labels)

    print(f"Extraction complete. Feature matrix: {X.shape} | dtype: {X.dtype}")
    print(f"Labels: {y.shape}")

    # Save with compression
    save_path = os.path.join(PROCESSED_DATA_PATH, "features.npz")
    np.savez_compressed(
        save_path,
        features=X,
        labels=y,
        label_names=np.array(label_names)
    )
    print(f"Saved compressed data → {save_path}")


if __name__ == "__main__":
    process_dataset(max_images_per_class=None, max_classes=None)

