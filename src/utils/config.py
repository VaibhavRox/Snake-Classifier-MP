
import os

# Base Paths — resolved relative to this file's location so the project is portable
# src/utils/config.py → go up 3 levels to reach the project root (Snake-Classifier-MP/)
_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))  # Snake-Classifier-MP/

DATASET_PATH       = os.environ.get("SNAKE_DATASET_PATH",   os.path.join(PROJECT_ROOT, "venomous_data"))
PROCESSED_DATA_PATH = os.environ.get("SNAKE_PROCESSED_PATH", os.path.join(PROJECT_ROOT, "data", "processed"))
ARTIFACTS_PATH     = os.environ.get("SNAKE_ARTIFACTS_PATH", os.path.join(PROJECT_ROOT, "src", "models", "artifacts"))

# Image Processing (for ResNet)
IMG_SIZE = (224, 224)  # ResNet input size

# Feature Extraction
FEATURE_EXTRACTOR = "vgg16"  # Options: "vgg16", "resnet50", "legacy" (HOG+LBP+HSV)
VGG_FEATURE_DIM = 4096       # Output dimension of VGG16 features
RESNET_FEATURE_DIM = 2048    # Output dimension of ResNet50 features (kept for reference)
BATCH_SIZE = 32              # Batch size for feature extraction

# Model Config
RANDOM_SEED = 42
TEST_SPLIT  = 0.2

# Augmentation (disabled - set to 0 for no augmentation)
AUGMENT_FACTOR = 0              # Augmented copies per image (0 = disabled)
MAX_AUG_RATIO = 0.0             # Max augmented:real ratio (0 = no augmentation)

# Legacy settings (kept for reference, not used with ResNet)
HOG_ORIENTATIONS     = 9
HOG_PIXELS_PER_CELL  = (8, 8)
HOG_CELLS_PER_BLOCK  = (2, 2)
LBP_POINTS = 16
LBP_RADIUS = 2
HSV_BINS = (8, 8, 8)
GAUSSIAN_BLUR_KERNEL = (5, 5)
