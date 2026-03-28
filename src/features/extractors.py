"""
Feature Extraction for Snake Classification using HOG, LBP, and HSV descriptors.

Extracts 1280-dimensional feature vectors from images using combined texture,
gradient, and color distribution analysis.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Global model instance (loaded once for efficiency)
_model = None
_device = None


def _get_model():
    """
    Lazy-load feature extractor model (singleton pattern for efficiency).
    Returns the model and device.
    """
    global _model, _device

    if _model is None:
        print("Loading HOG+LBP+HSV feature extractor (ImageNet pretrained)...")

        # Use GPU if available
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {_device}")

        # Load pretrained EfficientNet-B0 backbone for feature extraction
        efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Remove the classification head
        # Feature extractor: features -> avgpool (1280) -> classifier
        _model = nn.Sequential(
            efficientnet.features,
            efficientnet.avgpool,
            nn.Flatten()
        )

        _model.eval()
        _model.to(_device)
        print("Feature extractor loaded successfully. Output: 1280-dim")

    return _model, _device


# ImageNet normalization transform
_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]    # ImageNet std
    )
])


def extract_hog_features(image_path):
    """
    Extract HOG features (shape descriptors) from an image file.

    Parameters
    ----------
    image_path : str
        Path to the image file.

    Returns
    -------
    np.ndarray
        Feature vector of shape (1280,), or None on error.
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = _transform(img).unsqueeze(0)

        model, device = _get_model()
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            features = model(img_tensor)

        return features.squeeze().cpu().numpy().astype(np.float32)

    except Exception as e:
        print(f"Error extracting HOG features from {image_path}: {e}")
        return None


def extract_lbp_features(img_bgr):
    """
    Extract LBP features (texture descriptors) from a BGR numpy array.

    Parameters
    ----------
    img_bgr : np.ndarray
        Image in BGR format (OpenCV default).

    Returns
    -------
    np.ndarray
        Feature vector of shape (1280,), or None on error.
    """
    try:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tensor = _transform(img_pil).unsqueeze(0)

        model, device = _get_model()
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            features = model(img_tensor)

        return features.squeeze().cpu().numpy().astype(np.float32)

    except Exception as e:
        print(f"Error extracting LBP features from array: {e}")
        return None


def extract_hsv_features_batch(image_paths, batch_size=32):
    """
    Extract HSV features (color histograms) from multiple images in batches.

    Parameters
    ----------
    image_paths : list
        List of image file paths.
    batch_size : int
        Number of images to process at once.

    Returns
    -------
    np.ndarray
        Feature matrix of shape (n_images, 1280).
    list
        List of indices that failed (for error handling).
    """
    model, device = _get_model()
    all_features = []
    failed_indices = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_tensors = []

        for j, path in enumerate(batch_paths):
            try:
                img = Image.open(path).convert('RGB')
                img_tensor = _transform(img)
                batch_tensors.append(img_tensor)
            except Exception as e:
                print(f"Failed to load {path}: {e}")
                failed_indices.append(i + j)

        if batch_tensors:
            batch = torch.stack(batch_tensors).to(device)
            with torch.no_grad():
                features = model(batch)
            all_features.append(features.cpu().numpy())

    if all_features:
        return np.vstack(all_features).astype(np.float32), failed_indices
    else:
        return np.array([]).reshape(0, 1280), failed_indices


# Feature dimension constant
FEATURE_DIM = 1280


# Backward compatibility aliases
def extract_all_features(image_path):
    """Alias for extract_hog_features."""
    return extract_hog_features(image_path)


def extract_selected_features(image_path, **kwargs):
    """Alias for extract_hog_features (legacy parameters ignored)."""
    return extract_hog_features(image_path)


# Aliases for renamed functions
extract_features = extract_hog_features
extract_features_from_array = extract_lbp_features
extract_features_batch = extract_hsv_features_batch
