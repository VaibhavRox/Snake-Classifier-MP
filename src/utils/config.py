
import os

# Base Paths
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))

DATASET_PATH = os.environ.get("SNAKE_DATASET_PATH", os.path.join(PROJECT_ROOT, "data", "snake_reference_images"))
PROCESSED_DATA_PATH = os.environ.get("SNAKE_PROCESSED_PATH", os.path.join(PROJECT_ROOT, "data", "processed"))
ARTIFACTS_PATH = os.environ.get("SNAKE_ARTIFACTS_PATH", os.path.join(PROJECT_ROOT, "src", "models", "artifacts"))

# Image Processing
IMG_SIZE = (224, 224)

# Feature Extraction (HOG+LBP+HSV descriptors)
FEATURE_DIM = 1280
BATCH_SIZE = 32

# Model Training
RANDOM_SEED = 42
TEST_SPLIT = 0.2
CV_FOLDS = 5
TUNING_FOLDS = 3

# Models to train
MODEL_TYPES = ["linearsvc", "logreg"]

# Hyperparameter Grids
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
    }
}

# Augmentation (disabled)
AUGMENT_FACTOR = 0
MAX_AUG_RATIO = 0.0
