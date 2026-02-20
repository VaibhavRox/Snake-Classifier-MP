# 🐍 AI Snake Species Classifier

AI-powered snake species classifier (1,686 classes) using **Classical Machine Learning** — HOG + LBP + HSV features with LinearSVC or Logistic Regression. Includes venomous/non-venomous safety assessment and a Streamlit web UI.

## 🚀 Features

- **Multi-feature extraction** — HOG, LBP, HSV colour histograms (individually selectable)
- **No data leakage** — stratified split performed *before* scaling and PCA
- **LinearSVC / Logistic Regression / LightGBM** — no RandomForest memory crashes
- **Top-1 and Top-5 accuracy** reported after training
- **3-fold StratifiedKFold** cross-validation
- **Safety gate** — flags venomous species and low-confidence predictions as UNKNOWN
- **Streamlit UI** for real-time inference

---

## 🛠️ Setup (on any machine)

### 1. Clone
```bash
git clone https://github.com/VaibhavRox/Miniproject-AI-based-Snake-Classification.git
cd Miniproject-AI-based-Snake-Classification/Snake-Classifier-MP
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Point to your dataset
The dataset is **not** included in the repo (it's in `.gitignore`).
Tell the code where your dataset lives via an environment variable:

```bash
# macOS / Linux
export SNAKE_DATASET_PATH=/path/to/your/dataset

# Windows PowerShell
$env:SNAKE_DATASET_PATH = "C:\path\to\your\dataset"
```

Expected dataset structure:
```
<SNAKE_DATASET_PATH>/
├── species_name_1/
│   ├── img001.jpg
│   └── ...
├── species_name_2/
│   └── ...
└── ...
```

---

## 🏃 Training

```bash
python -m src.models.train
```

This runs the full pipeline:
```
Load data → Stratified split (80/20) → StandardScaler → PCA (300 components)
→ 3-fold CV → Train LinearSVC → Top-1 / Top-5 accuracy → Save artifacts
```

Artifacts are saved to `src/models/artifacts/`:
- `scaler.pkl`
- `pca.pkl`
- `model.pkl`
- `label_names.pkl`

### Switching models or feature subsets
Edit the config block at the bottom of `src/models/train.py`:

```python
MODEL_TYPE   = "linearsvc"   # "linearsvc" | "logreg" | "lgbm"
N_COMPONENTS = 300           # PCA: try 200 / 300 / 500
RUN_CV       = True

USE_HOG = True   # disable any to experiment with subsets
USE_LBP = True
USE_HSV = True
```

---

## 🌐 Running the Web App

> Requires trained artifacts in `src/models/artifacts/` first.

```bash
streamlit run src/app.py
```

## 🧪 Quick Inference Test

```bash
python test_inference.py
```

---

## 📂 Project Structure

```
Snake-Classifier-MP/
├── data/
│   ├── dataset/            # ← put your dataset here (or set env var)
│   └── processed/          # extracted features saved as features.npz
├── src/
│   ├── features/
│   │   ├── extractors.py   # HOG / LBP / HSV extraction (float32)
│   │   └── pipeline.py     # batch extraction + compressed save
│   ├── models/
│   │   ├── train.py        # full training pipeline
│   │   └── artifacts/      # scaler.pkl, pca.pkl, model.pkl (git-ignored)
│   ├── utils/
│   │   ├── config.py       # all paths & hyperparameters
│   │   └── safety.py       # venomous species detection
│   ├── app.py              # Streamlit UI
│   └── inference.py        # SnakeClassifier inference class
├── test_inference.py
├── requirements.txt
└── README.md
```

---

## ⚠️ Disclaimer

This tool is for educational and experimental purposes only. **Do not rely solely on this AI in real dangerous situations.** Always verify with expert herpetological knowledge.
