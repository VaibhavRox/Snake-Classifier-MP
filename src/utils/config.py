
import os

# Base Paths — resolved relative to this file's location so the project is portable
# src/utils/config.py → go up 3 levels to reach the project root (Snake-Classifier-MP/)
_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))  # Snake-Classifier-MP/

DATASET_PATH       = os.environ.get("SNAKE_DATASET_PATH",   os.path.join(PROJECT_ROOT, "dataset"))
PROCESSED_DATA_PATH = os.environ.get("SNAKE_PROCESSED_PATH", os.path.join(PROJECT_ROOT, "data", "processed"))
ARTIFACTS_PATH     = os.environ.get("SNAKE_ARTIFACTS_PATH", os.path.join(PROJECT_ROOT, "src", "models", "artifacts"))

# Image Processing
IMG_SIZE = (256, 256)       # Resize target
GAUSSIAN_BLUR_KERNEL = (5, 5)

# Feature Config
HOG_ORIENTATIONS     = 9
HOG_PIXELS_PER_CELL  = (8, 8)
HOG_CELLS_PER_BLOCK  = (2, 2)

LBP_POINTS = 24   # Number of circular neighbourhood points
LBP_RADIUS = 3    # Radius of LBP circle

HSV_BINS = (8, 8, 8)  # Hue, Saturation, Value histogram bins

# Model Config
RANDOM_SEED = 42
TEST_SPLIT  = 0.2

# PCA
PCA_COMPONENTS = 1000  # Increased for better variance preservation
