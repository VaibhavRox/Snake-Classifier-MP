"""
Deep Learning Feature Extraction for Snake Classification.

Supports multiple pretrained models:
- ResNet50: 2048-dimensional features
- EfficientNet-B0: 1280-dimensional features

All models use ImageNet pretrained weights with Global Average Pooling.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Global model instances (lazy-loaded singletons)
_resnet_model = None
_efficientnet_model = None
_device = None


def _get_device():
    """Get the compute device (GPU if available, else CPU)."""
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {_device}")
    return _device


# =============================================================================
# ResNet50 Feature Extractor (2048-dim)
# =============================================================================

def _get_resnet_model():
    """
    Lazy-load ResNet50 model (singleton pattern for efficiency).
    Returns the model and device.
    """
    global _resnet_model
    device = _get_device()

    if _resnet_model is None:
        print("Loading ResNet50 (ImageNet pretrained)...")

        # Load pretrained ResNet50
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Remove the classification head (fc layer)
        # ResNet50 architecture: conv layers -> avgpool (2048) -> fc (1000)
        _resnet_model = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
            nn.Flatten()
        )

        _resnet_model.eval()
        _resnet_model.to(device)
        print("ResNet50 loaded successfully. Output: 2048-dim")

    return _resnet_model, device


# =============================================================================
# EfficientNet-B0 Feature Extractor (1280-dim)
# =============================================================================

def _get_efficientnet_model():
    """
    Lazy-load EfficientNet-B0 model (singleton pattern for efficiency).
    Returns the model and device.
    """
    global _efficientnet_model
    device = _get_device()

    if _efficientnet_model is None:
        print("Loading EfficientNet-B0 (ImageNet pretrained)...")

        # Load pretrained EfficientNet-B0
        efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Remove the classification head
        # EfficientNet-B0 architecture: features -> avgpool (1280) -> classifier
        _efficientnet_model = nn.Sequential(
            efficientnet.features,
            efficientnet.avgpool,
            nn.Flatten()
        )

        _efficientnet_model.eval()
        _efficientnet_model.to(device)
        print("EfficientNet-B0 loaded successfully. Output: 1280-dim")

    return _efficientnet_model, device


# =============================================================================
# ImageNet Normalization Transform (shared by all models)
# =============================================================================

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]    # ImageNet std
    )
])


# =============================================================================
# ResNet50 Feature Extraction Functions
# =============================================================================

def extract_resnet_features(image_path):
    """
    Extract ResNet50 features from an image file.

    Parameters
    ----------
    image_path : str
        Path to the image file.

    Returns
    -------
    np.ndarray
        Feature vector of shape (2048,), or None on error.
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = _transform(img).unsqueeze(0)

        model, device = _get_resnet_model()
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            features = model(img_tensor)

        return features.squeeze().cpu().numpy().astype(np.float32)

    except Exception as e:
        print(f"Error extracting ResNet features from {image_path}: {e}")
        return None


def extract_resnet_features_from_array(img_bgr):
    """
    Extract ResNet50 features from a BGR numpy array (OpenCV format).

    Parameters
    ----------
    img_bgr : np.ndarray
        Image in BGR format (OpenCV default).

    Returns
    -------
    np.ndarray
        Feature vector of shape (2048,), or None on error.
    """
    try:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tensor = _transform(img_pil).unsqueeze(0)

        model, device = _get_resnet_model()
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            features = model(img_tensor)

        return features.squeeze().cpu().numpy().astype(np.float32)

    except Exception as e:
        print(f"Error extracting ResNet features from array: {e}")
        return None


def extract_resnet_features_batch(image_paths, batch_size=32):
    """
    Extract ResNet50 features from multiple images in batches.

    Parameters
    ----------
    image_paths : list
        List of image file paths.
    batch_size : int
        Number of images to process at once.

    Returns
    -------
    np.ndarray
        Feature matrix of shape (n_images, 2048).
    list
        List of indices that failed (for error handling).
    """
    model, device = _get_resnet_model()
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
        return np.array([]).reshape(0, 2048), failed_indices


# =============================================================================
# EfficientNet-B0 Feature Extraction Functions
# =============================================================================

def extract_efficientnet_features(image_path):
    """
    Extract EfficientNet-B0 features from an image file.

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

        model, device = _get_efficientnet_model()
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            features = model(img_tensor)

        return features.squeeze().cpu().numpy().astype(np.float32)

    except Exception as e:
        print(f"Error extracting EfficientNet features from {image_path}: {e}")
        return None


def extract_efficientnet_features_from_array(img_bgr):
    """
    Extract EfficientNet-B0 features from a BGR numpy array (OpenCV format).

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

        model, device = _get_efficientnet_model()
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            features = model(img_tensor)

        return features.squeeze().cpu().numpy().astype(np.float32)

    except Exception as e:
        print(f"Error extracting EfficientNet features from array: {e}")
        return None


def extract_efficientnet_features_batch(image_paths, batch_size=32):
    """
    Extract EfficientNet-B0 features from multiple images in batches.

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
    model, device = _get_efficientnet_model()
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


# =============================================================================
# Generic Feature Extraction (supports multiple extractors)
# =============================================================================

def extract_features(image_path, extractor="efficientnet"):
    """
    Extract features using the specified extractor.

    Parameters
    ----------
    image_path : str
        Path to the image file.
    extractor : str
        Feature extractor to use: "resnet50" or "efficientnet"

    Returns
    -------
    np.ndarray
        Feature vector, or None on error.
    """
    extractor = extractor.lower()
    if extractor in ["efficientnet", "efficientnet_b0", "efficientnetb0"]:
        return extract_efficientnet_features(image_path)
    elif extractor in ["resnet", "resnet50"]:
        return extract_resnet_features(image_path)
    else:
        raise ValueError(f"Unknown extractor: {extractor}. Use 'resnet50' or 'efficientnet'")


def extract_features_from_array(img_bgr, extractor="efficientnet"):
    """
    Extract features from a BGR numpy array using the specified extractor.

    Parameters
    ----------
    img_bgr : np.ndarray
        Image in BGR format (OpenCV default).
    extractor : str
        Feature extractor to use: "resnet50" or "efficientnet"

    Returns
    -------
    np.ndarray
        Feature vector, or None on error.
    """
    extractor = extractor.lower()
    if extractor in ["efficientnet", "efficientnet_b0", "efficientnetb0"]:
        return extract_efficientnet_features_from_array(img_bgr)
    elif extractor in ["resnet", "resnet50"]:
        return extract_resnet_features_from_array(img_bgr)
    else:
        raise ValueError(f"Unknown extractor: {extractor}. Use 'resnet50' or 'efficientnet'")


def extract_features_batch(image_paths, extractor="efficientnet", batch_size=32):
    """
    Extract features from multiple images using the specified extractor.

    Parameters
    ----------
    image_paths : list
        List of image file paths.
    extractor : str
        Feature extractor to use: "resnet50" or "efficientnet"
    batch_size : int
        Number of images to process at once.

    Returns
    -------
    np.ndarray
        Feature matrix of shape (n_images, feature_dim).
    list
        List of indices that failed.
    """
    extractor = extractor.lower()
    if extractor in ["efficientnet", "efficientnet_b0", "efficientnetb0"]:
        return extract_efficientnet_features_batch(image_paths, batch_size)
    elif extractor in ["resnet", "resnet50"]:
        return extract_resnet_features_batch(image_paths, batch_size)
    else:
        raise ValueError(f"Unknown extractor: {extractor}. Use 'resnet50' or 'efficientnet'")


# =============================================================================
# Feature Dimension Constants
# =============================================================================

FEATURE_DIMS = {
    "resnet50": 2048,
    "efficientnet": 1280,
    "efficientnet_b0": 1280,
}


def get_feature_dim(extractor):
    """Get the output dimension for a given extractor."""
    extractor = extractor.lower()
    if extractor in ["efficientnet", "efficientnet_b0", "efficientnetb0"]:
        return 1280
    elif extractor in ["resnet", "resnet50"]:
        return 2048
    else:
        raise ValueError(f"Unknown extractor: {extractor}")


# =============================================================================
# Backward Compatibility Aliases
# =============================================================================

def extract_all_features(image_path):
    """Alias for extract_efficientnet_features (default extractor)."""
    return extract_efficientnet_features(image_path)


def extract_selected_features(image_path, use_hog=True, use_lbp=True, use_hsv=True):
    """Alias for extract_efficientnet_features (legacy parameters ignored)."""
    return extract_efficientnet_features(image_path)


# =============================================================================
# Legacy Functions
# =============================================================================

def preprocess_image(image_path):
    """Legacy: Load and resize image."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    return cv2.resize(img, (224, 224))


def preprocess_image_array(img_bgr):
    """Legacy: Resize image array."""
    return cv2.resize(img_bgr, (224, 224))
