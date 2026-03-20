"""
VGG16-based Feature Extraction for Snake Classification.

Uses pretrained VGG16 (ImageNet weights) to extract 4096-dimensional
feature vectors from images. This replaces the previous ResNet50 approach.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Global model instance (loaded once for efficiency)
_vgg_model = None
_device = None


def _get_vgg_model():
    """
    Lazy-load VGG16 model (singleton pattern for efficiency).
    Returns the model and device.
    """
    global _vgg_model, _device

    if _vgg_model is None:
        print("Loading VGG16 (ImageNet pretrained)...")

        # Use GPU if available
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {_device}")

        # Load pretrained VGG16
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        # Use features (conv layers) + avgpool + first part of classifier
        # VGG classifier: Linear(25088, 4096) -> ReLU -> Dropout -> Linear(4096, 4096) -> ReLU -> Dropout -> Linear(4096, 1000)
        # We want the 4096-dim output after the first FC + ReLU
        _vgg_model = nn.Sequential(
            vgg.features,
            vgg.avgpool,
            nn.Flatten(),
            vgg.classifier[0],  # Linear(25088, 4096)
            vgg.classifier[1],  # ReLU
        )

        # Set to eval mode and move to device
        _vgg_model.eval()
        _vgg_model.to(_device)

        print("VGG16 loaded successfully.")

    return _vgg_model, _device


# ImageNet normalization transform
_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]    # ImageNet std
    )
])


def extract_resnet_features(image_path):
    """
    Extract VGG16 features from an image file.

    Parameters
    ----------
    image_path : str
        Path to the image file.

    Returns
    -------
    np.ndarray
        Feature vector of shape (4096,), or None on error.
    """
    try:
        # Load image using PIL (RGB format)
        img = Image.open(image_path).convert('RGB')

        # Apply transforms
        img_tensor = _transform(img).unsqueeze(0)  # Add batch dimension

        # Get model and device
        model, device = _get_vgg_model()
        img_tensor = img_tensor.to(device)

        # Extract features (no gradients needed)
        with torch.no_grad():
            features = model(img_tensor)

        # Flatten and convert to numpy
        features = features.squeeze().cpu().numpy()

        return features.astype(np.float32)

    except Exception as e:
        print(f"Error extracting features from {image_path}: {e}")
        return None


def extract_resnet_features_from_array(img_bgr):
    """
    Extract VGG16 features from a BGR numpy array (OpenCV format).

    Parameters
    ----------
    img_bgr : np.ndarray
        Image in BGR format (OpenCV default).

    Returns
    -------
    np.ndarray
        Feature vector of shape (4096,), or None on error.
    """
    try:
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        img_pil = Image.fromarray(img_rgb)

        # Apply transforms
        img_tensor = _transform(img_pil).unsqueeze(0)

        # Get model and device
        model, device = _get_vgg_model()
        img_tensor = img_tensor.to(device)

        # Extract features
        with torch.no_grad():
            features = model(img_tensor)

        # Flatten and convert to numpy
        features = features.squeeze().cpu().numpy()

        return features.astype(np.float32)

    except Exception as e:
        print(f"Error extracting features from array: {e}")
        return None


def extract_resnet_features_batch(image_paths, batch_size=32):
    """
    Extract VGG16 features from multiple images in batches.

    Parameters
    ----------
    image_paths : list
        List of image file paths.
    batch_size : int
        Number of images to process at once.

    Returns
    -------
    np.ndarray
        Feature matrix of shape (n_images, 4096).
    list
        List of indices that failed (for error handling).
    """
    model, device = _get_vgg_model()

    all_features = []
    failed_indices = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_tensors = []
        batch_indices = []

        for j, path in enumerate(batch_paths):
            try:
                img = Image.open(path).convert('RGB')
                img_tensor = _transform(img)
                batch_tensors.append(img_tensor)
                batch_indices.append(i + j)
            except Exception as e:
                print(f"Failed to load {path}: {e}")
                failed_indices.append(i + j)

        if batch_tensors:
            # Stack into batch
            batch = torch.stack(batch_tensors).to(device)

            # Extract features
            with torch.no_grad():
                features = model(batch)

            # Convert to numpy (already flattened by VGG model)
            features = features.cpu().numpy()

            all_features.append(features)

    if all_features:
        return np.vstack(all_features).astype(np.float32), failed_indices
    else:
        return np.array([]).reshape(0, 4096), failed_indices


# Backward compatibility aliases
def extract_all_features(image_path):
    """Alias for extract_resnet_features (backward compatibility)."""
    return extract_resnet_features(image_path)


def extract_features_from_array(img_bgr, use_hog=True, use_lbp=True, use_hsv=True):
    """
    Alias for extract_resnet_features_from_array (backward compatibility).
    The use_hog, use_lbp, use_hsv parameters are ignored.
    """
    return extract_resnet_features_from_array(img_bgr)


def extract_selected_features(image_path, use_hog=True, use_lbp=True, use_hsv=True):
    """
    Alias for extract_resnet_features (backward compatibility).
    The use_hog, use_lbp, use_hsv parameters are ignored.
    """
    return extract_resnet_features(image_path)


# Legacy functions (kept for reference, but not used)
def preprocess_image(image_path):
    """Legacy: Load and resize image."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    return cv2.resize(img, (224, 224))


def preprocess_image_array(img_bgr):
    """Legacy: Resize image array."""
    return cv2.resize(img_bgr, (224, 224))
