
import os

# Base Paths — resolved relative to this file's location so the project is portable
# src/utils/config.py → go up 3 levels to reach the project root (Snake-Classifier-MP/)
_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))  # Snake-Classifier-MP/

DATASET_PATH       = os.environ.get("SNAKE_DATASET_PATH",   os.path.join(PROJECT_ROOT, "venomous_data"))
PROCESSED_DATA_PATH = os.environ.get("SNAKE_PROCESSED_PATH", os.path.join(PROJECT_ROOT, "data", "processed"))
ARTIFACTS_PATH     = os.environ.get("SNAKE_ARTIFACTS_PATH", os.path.join(PROJECT_ROOT, "src", "models", "artifacts"))

# Image Processing
IMG_SIZE = (224, 224)  # Input size for all models

# =============================================================================
# Feature Extraction Configuration
# =============================================================================

# Default feature extractor: "efficientnet" or "resnet50"
FEATURE_EXTRACTOR = "efficientnet"

# Feature dimensions for each extractor
EFFICIENTNET_FEATURE_DIM = 1280   # EfficientNet-B0 output
RESNET_FEATURE_DIM = 2048         # ResNet50 output
VGG_FEATURE_DIM = 4096            # VGG16 output (legacy)

# Batch size for feature extraction
BATCH_SIZE = 32

# Available extractors for comparison
AVAILABLE_EXTRACTORS = ["resnet50", "efficientnet"]

# =============================================================================
# Model Training Configuration
# =============================================================================

RANDOM_SEED = 42
TEST_SPLIT  = 0.2

# Cross-validation settings
CV_FOLDS = 5           # Folds for final evaluation
TUNING_FOLDS = 3       # Folds for hyperparameter tuning

# Models to train
MODEL_TYPES = ["linearsvc", "logreg", "lgbm"]

# =============================================================================
# Hyperparameter Grids for Tuning
# =============================================================================

PARAM_GRIDS = {
    'linearsvc': {
        'C': [0.1, 1, 5, 10],
        'loss': ['hinge', 'squared_hinge'],
    },
    'logreg': {
        'C': [0.1, 1, 5, 10],
        'solver': ['lbfgs', 'saga'],
        'penalty': ['l2'],
        'max_iter': [2000],
    },
    'lgbm': {
        'n_estimators': [100, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [15, 31, 63],
        'max_depth': [-1, 5, 10],
    }
}

# =============================================================================
# Augmentation (disabled by default)
# =============================================================================

AUGMENT_FACTOR = 0              # Augmented copies per image (0 = disabled)
MAX_AUG_RATIO = 0.0             # Max augmented:real ratio (0 = no augmentation)

# =============================================================================
# PCA Settings (DISABLED - not recommended for deep features)
# =============================================================================

PCA_COMPONENTS = None  # Set to None to disable PCA

# =============================================================================
# Legacy settings (kept for reference, not used)
# =============================================================================

HOG_ORIENTATIONS     = 9
HOG_PIXELS_PER_CELL  = (8, 8)
HOG_CELLS_PER_BLOCK  = (2, 2)
LBP_POINTS = 16
LBP_RADIUS = 2
HSV_BINS = (8, 8, 8)
GAUSSIAN_BLUR_KERNEL = (5, 5)
