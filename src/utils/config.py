
import os

# Base Paths
PROJECT_ROOT = "g:/Miniproject"
DATASET_PATH = os.path.join(PROJECT_ROOT, "Miniproject-Dataset")
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data/processed")

# Image Processing
IMG_SIZE = (128, 128)  # Resize target
GAUSSIAN_BLUR_KERNEL = (5, 5)

# Feature Config
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)

LBP_POINTS = 24  # Number of points
LBP_RADIUS = 3   # Radius of circle

HSV_BINS = (8, 8, 8) # Hue, Saturation, Value bins

# Model Config
RANDOM_SEED = 42
TEST_SPLIT = 0.2
