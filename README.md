# AI Snake Species Classifier

AI-powered snake species classifier (1,686 classes) using classical machine learning.
The pipeline combines HOG + LBP + HSV features, applies StandardScaler + PCA,
and supports LinearSVC and Logistic Regression.

## Features

- Configurable feature extraction (HOG, LBP, HSV in any combination)
- Online image augmentation during feature extraction
	(random flip, 90-degree rotation, brightness jitter)
- No data leakage (stratified split before scaler/PCA fit)
- Optional 3-fold StratifiedKFold cross-validation
- Top-1 and Top-5 evaluation metrics
- Venom safety-aware postprocessing for inference output
- Streamlit UI for interactive predictions

## Setup

### 1. Clone

```bash
git clone https://github.com/VaibhavRox/Snake-Classifier-MP.git
cd Snake-Classifier-MP
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure paths (optional but recommended)

All paths are configurable via environment variables in `src/utils/config.py`.

```bash
# Dataset root (contains species folders)
# Default: <project_root>/dataset
export SNAKE_DATASET_PATH=/path/to/dataset

# Processed features output folder
# Default: <project_root>/data/processed
export SNAKE_PROCESSED_PATH=/path/to/processed

# Artifacts folder used by inference
# Default: <project_root>/src/models/artifacts
export SNAKE_ARTIFACTS_PATH=/path/to/artifacts
```

Windows PowerShell example:

```powershell
$env:SNAKE_DATASET_PATH = "C:\path\to\dataset"
$env:SNAKE_PROCESSED_PATH = "C:\path\to\processed"
$env:SNAKE_ARTIFACTS_PATH = "C:\path\to\artifacts"
```

Expected dataset layout:

```text
<SNAKE_DATASET_PATH>/
	Species_A/
		img001.jpg
		...
	Species_B/
		...
```

## Training

Run:

```bash
python -m src.models.train
```

Current training flow in `src/models/train.py`:

1. Load compressed feature matrix (`features.npz`) or build it with `src/features/pipeline.py`
2. Stratified train/test split (80/20)
3. StandardScaler fit on train only
4. PCA projection (configured in code)
5. Optional CV (3-fold)
6. Train models listed in `MODEL_TYPES`
7. Save artifacts for each model

By default, training iterates through all model types:

- `linearsvc`
- `logreg`

Artifacts are saved per model subfolder:

```text
src/models/artifacts/
	linearsvc/
		scaler.pkl
		pca.pkl
		model.pkl
		label_names.pkl
	logreg/
		...
```

### Important for inference

`SnakeClassifier` expects one set of artifacts in the directory pointed to by
`SNAKE_ARTIFACTS_PATH`. Since training now saves subfolders per model,
set the variable to a specific model folder before running inference:

```powershell
$env:SNAKE_ARTIFACTS_PATH = "C:\...\Snake-Classifier-MP\src\models\artifacts\linearsvc"
```

## Run Inference

### Streamlit app

```bash
streamlit run src/app.py
```

### CLI smoke test

```bash
python test_inference.py
```

## Project Structure

```text
Snake-Classifier-MP/
	data/
		processed/
	dataset/
	src/
		features/
			augmentation.py   # random flip/rotate/brightness augmentation
			extractors.py     # HOG/LBP/HSV feature extraction
			pipeline.py       # dataset -> compressed features.npz
		models/
			train.py          # full train/eval/save pipeline
			artifacts/        # model-wise artifacts (gitignored)
		utils/
			config.py         # paths + hyperparameters
			safety.py         # venom safety checks
		inference.py        # SnakeClassifier class
		app.py              # Streamlit UI
	test_inference.py
	requirements.txt
	README.md
```

## Notes on Large Files

- `*.pkl` files are ignored by `.gitignore`.
- Do not commit model artifacts to GitHub.
- If large artifact blobs were committed before being ignored, remove them from
	commit history before pushing.

## Disclaimer

This project is for educational and experimental use.
Do not rely on model output alone in dangerous real-world snake encounters.
