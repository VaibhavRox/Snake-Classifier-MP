import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from src.utils.config import (
    IMG_SIZE,
    HOG_ORIENTATIONS, HOG_PIXELS_PER_CELL, HOG_CELLS_PER_BLOCK,
    LBP_POINTS, LBP_RADIUS,
    HSV_BINS
)

def preprocess_image(image_path):
    """
    Reads an image and resizes it.
    NOTE: Augmentation removed - should be applied separately at pipeline level.
    Returns the processed image in BGR (OpenCV default).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    img = cv2.resize(img, IMG_SIZE)
    return img


def preprocess_image_array(img_bgr):
    """
    Preprocesses an image array (already loaded).
    Used for augmented images passed as numpy arrays.
    """
    if img_bgr is None:
        raise ValueError("Image array is None")
    img = cv2.resize(img_bgr, IMG_SIZE)
    return img


def extract_hog(img_gray):
    """
    Extracts Histogram of Oriented Gradients (HOG) features.
    Input : Grayscale image (uint8 or float)
    Output: 1-D float32 array
    """
    features = hog(
        img_gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm='L2-Hys',
        visualize=False
    )
    return features.astype(np.float32)


def extract_lbp(img_gray):
    """
    Extracts Local Binary Patterns (LBP) histogram.
    Input : Grayscale image
    Output: 1-D float32 array
    """
    lbp = local_binary_pattern(img_gray, LBP_POINTS, LBP_RADIUS, method='uniform')
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float32)


def extract_color_histogram(img_bgr):
    """
    Extracts a colour histogram in HSV space (concatenated per-channel).
    Input : BGR image (OpenCV default)
    Output: 1-D float32 array
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, HSV_BINS, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten().astype(np.float32)


def extract_all_features(image_path):
    """
    Convenience wrapper: concatenates HOG + LBP + HSV features.
    Returns a 1-D float32 array, or None on error.
    """
    return extract_selected_features(image_path, use_hog=True, use_lbp=True, use_hsv=True)


def extract_selected_features(image_path, use_hog=True, use_lbp=True, use_hsv=True):
    """
    Extracts a configurable subset of features (HOG / LBP / HSV).
    At least one feature type must be enabled.
    Returns a 1-D float32 array, or None on error.
    """
    if not any([use_hog, use_lbp, use_hsv]):
        raise ValueError("At least one feature type (HOG / LBP / HSV) must be enabled.")

    try:
        img_bgr  = preprocess_image(image_path)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        parts = []
        if use_hog:
            parts.append(extract_hog(img_gray))
        if use_lbp:
            parts.append(extract_lbp(img_gray))
        if use_hsv:
            parts.append(extract_color_histogram(img_bgr))

        return np.hstack(parts).astype(np.float32)

    except Exception as e:
        print(f"Error extracting features from {image_path}: {e}")
        return None


def extract_features_from_array(img_bgr, use_hog=True, use_lbp=True, use_hsv=True):
    """
    Extracts features from an image array (already loaded as BGR numpy array).
    Used for augmented images that are generated in-memory.
    Returns a 1-D float32 array, or None on error.
    """
    if not any([use_hog, use_lbp, use_hsv]):
        raise ValueError("At least one feature type (HOG / LBP / HSV) must be enabled.")

    try:
        img_bgr  = preprocess_image_array(img_bgr)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        parts = []
        if use_hog:
            parts.append(extract_hog(img_gray))
        if use_lbp:
            parts.append(extract_lbp(img_gray))
        if use_hsv:
            parts.append(extract_color_histogram(img_bgr))

        return np.hstack(parts).astype(np.float32)

    except Exception as e:
        print(f"Error extracting features from image array: {e}")
        return None

