
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
    Reads an image, resizes it, and applies Gaussian blur.
    Returns the processed image in BGR (for OpenCV) or RGB (for others).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Resize
    img = cv2.resize(img, IMG_SIZE)
    
    # Noise Reduction
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    return img

def extract_hog(img_gray):
    """
    Extracts Histogram of Oriented Gradients (HOG) features.
    Input: Grayscale Image
    """
    features = hog(
        img_gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm='L2-Hys',
        visualize=False
    )
    return features

def extract_lbp(img_gray):
    """
    Extracts Local Binary Patterns (LBP) histogram.
    Input: Grayscale Image
    """
    lbp = local_binary_pattern(img_gray, LBP_POINTS, LBP_RADIUS, method='uniform')
    
    # Calculate histogram of LBP codes
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    
    return hist

def extract_color_histogram(img_bgr):
    """
    Extracts Color Histogram in HSV space.
    Input: BGR Image (OpenCV default)
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # Calculate histogram for each channel
    # Concatenate standard histograms
    # Using 8 bins per channel as defined in config
    hist = cv2.calcHist([hsv], [0, 1, 2], None, HSV_BINS, [0, 180, 0, 256, 0, 256])
    
    # Normalize and flatten
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_all_features(image_path):
    """
    Master function to get the concatenated feature vector.
    """
    try:
        img_bgr = preprocess_image(image_path)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        feat_hog = extract_hog(img_gray)
        feat_lbp = extract_lbp(img_gray)
        feat_color = extract_color_histogram(img_bgr)
        
        # Concatenate
        combined_features = np.hstack([feat_hog, feat_lbp, feat_color])
        return combined_features
    except Exception as e:
        print(f"Error extracting features from {image_path}: {e}")
        return None
