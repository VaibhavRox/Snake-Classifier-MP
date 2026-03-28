# 🐍 AI Snake Species Classifier

A production-ready snake species identification system using **HOG, LBP, and HSV** feature descriptors and **Logistic Regression** classification.

## ✨ Features

- **Classical Feature Extraction**: HOG (shape) + LBP (texture) + HSV (color) - 1280-dimensional feature vectors
- **High Accuracy**: 66.5% Top-1, 90.5% Top-3, 97.5% Top-5
- **Safety-Aware**: Venomous detection with confidence thresholds
- **Web UI**: Professional Streamlit web application
- **Reference Images**: Visual comparison with training data
- **Multiple Models**: Support for Logistic Regression & LinearSVC

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/VaibhavRox/Snake-Classifier-MP.git
cd Snake-Classifier-MP
```

### 2. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Application

```bash
streamlit run app/streamlit_app.py
```

The app will be available at: **http://localhost:8501**

### 4. Verify Installation

```bash
python test_setup.py
```

Expected output: `6/6 tests passed`

## 📁 Project Structure

```
Snake-Classifier-MP/
├── app/
│   └── streamlit_app.py          # Main web application (UI)
│
├── src/
│   ├── inference/
│   │   └── predictor.py          # SnakeClassifier (model loading & prediction)
│   │
│   ├── features/
│   │   ├── extractors.py         # HOG+LBP+HSV feature extraction
│   │   ├── pipeline.py           # Dataset processing pipeline
│   │   └── augmentation.py       # Image augmentation utilities
│   │
│   ├── models/
│   │   ├── train.py              # Model training script
│   │   └── artifacts/
│   │       ├── logreg/           # Logistic Regression (67% acc)
│   │       └── linearsvc/        # LinearSVC (62% acc)
│   │
│   └── utils/
│       ├── config.py             # Configuration & paths
│       └── safety.py             # Venomous detection & safety logic
│
├── data/
│   ├── raw/                      # Raw data (future use)
│   ├── processed/                # Cached feature vectors
│   └── snake_reference_images/   # Reference images (10 species)
│
├── notebooks/
│   └── inference_evaluation.ipynb  # Model evaluation & metrics
│
├── requirements.txt              # Python dependencies
├── test_setup.py                 # System verification script
└── README.md                     # This file
```

## 🎯 Usage

### Web Application

1. **Start the app**:
   ```bash
   streamlit run app/streamlit_app.py
   ```

2. **Upload snake image**: Drag and drop or browse for an image

3. **View predictions**: See top-3 predicted species with confidence scores

4. **Check reference images**: Compare with training data visuals

5. **Follow safety guidelines**: Read venomous/non-venomous warnings

### Python API

```python
from src.inference.predictor import SnakeClassifier

# Load classifier
classifier = SnakeClassifier(model_type="logreg")

# Run prediction
result = classifier.predict("path/to/snake_image.jpg")

# Access results
print(result["top_3"])           # Top 3 predictions
print(result["safety_message"])  # Safety assessment
```

## 🔬 Technical Details

### ML Pipeline

1. **Input**: User uploads image (any resolution)
2. **Preprocessing**: Resize to 224×224, ImageNet normalization
3. **Feature Extraction**: HOG+LBP+HSV → 1280-dim features
4. **Scaling**: StandardScaler (fitted on training data)
5. **Classification**: Logistic Regression (One-vs-Rest)
6. **Output**: Top-3 species with probabilities

### Model Performance

| Model | Top-1 | Top-3 | Top-5 |
|-------|-------|-------|-------|
| **Logistic Regression** | 66.5% | 90.5% | 97.5% |
| LinearSVC | 62.0% | 83.0% | 91.0% |

### Dataset

- **Species**: 10 snake species
- **Images**: 1,000 total (100 per species)
- **Split**: 80% train, 20% test
- **Classes**: Venomous and non-venomous species

## 📝 Configuration

### Environment Variables (Optional)

Configure paths via environment variables:

```bash
# Dataset path (reference images)
export SNAKE_DATASET_PATH=/path/to/data/snake_reference_images

# Processed features cache
export SNAKE_PROCESSED_PATH=/path/to/data/processed

# Model artifacts
export SNAKE_ARTIFACTS_PATH=/path/to/src/models/artifacts
```

**Windows PowerShell**:
```powershell
$env:SNAKE_DATASET_PATH = "C:\path\to\data\snake_reference_images"
$env:SNAKE_PROCESSED_PATH = "C:\path\to\data\processed"
$env:SNAKE_ARTIFACTS_PATH = "C:\path\to\src\models\artifacts"
```

**Default paths** (if not set):
- Dataset: `PROJECT_ROOT/data/snake_reference_images`
- Processed: `PROJECT_ROOT/data/processed`
- Artifacts: `PROJECT_ROOT/src/models/artifacts`

## 🔧 Training New Models

### Extract Features

```bash
python -c "from src.features.pipeline import process_dataset; process_dataset()"
```

### Train Models

```bash
python src/models/train.py
```

This will:
1. Load or extract features using HOG+LBP+HSV descriptors
2. Split data (80/20 train/test)
3. Apply StandardScaler
4. Perform hyperparameter tuning with GridSearchCV
5. Train Logistic Regression and LinearSVC
6. Evaluate models (Top-1, Top-3, Top-5 accuracy)
7. Save best models to `src/models/artifacts/`

## 📊 Model Artifacts

Each trained model saves:
```
src/models/artifacts/<model_type>/
├── model.pkl         # Trained classifier
├── scaler.pkl        # StandardScaler
└── label_names.pkl   # Species names
```

## ⚠️ Safety Features

### Confidence Threshold

- **Default**: 40% minimum confidence
- **Behavior**: If all predictions < 40%, shows "UNKNOWN" warning
- **Purpose**: Prevent dangerous misidentification

### Venomous Detection

- Automatic detection based on scientific names
- Color-coded warnings (red for venomous, green for safe)
- Safety guidelines displayed for each prediction

## 🧪 Testing

Run comprehensive tests:

```bash
python test_setup.py
```

Tests verify:
- ✅ All imports work
- ✅ Project structure is correct
- ✅ Model artifacts exist
- ✅ Reference images accessible
- ✅ Classifier loads successfully
- ✅ Feature extraction works

## 📚 Documentation

- **QUICKSTART.md** - Quick start guide with examples
- **IMPORT_FIX.md** - Details on import path configuration
- **test_setup.py** - Automated verification script

## 🐛 Troubleshooting

### ModuleNotFoundError: No module named 'src'

**Solution**: The `app/streamlit_app.py` file automatically adds the project root to Python path. If you still see this error:

1. Ensure you're running from the project root:
   ```bash
   cd Snake-Classifier-MP
   streamlit run app/streamlit_app.py
   ```

2. Verify the project structure is correct:
   ```bash
   python test_setup.py
   ```

See **IMPORT_FIX.md** for detailed explanation.

### No reference images shown

**Issue**: Folder name doesn't match species name

**Solution**: Ensure folder names in `data/snake_reference_images/` exactly match species names (e.g., `Naja_naja`)

### Model not loaded

**Issue**: Missing model artifacts

**Solution**:
1. Check that `src/models/artifacts/logreg/` contains `.pkl` files
2. Retrain models if needed: `python src/models/train.py`

## 🌟 Features in Detail

### Web Application

- **Clean UI**: Professional design with custom CSS
- **Model Selection**: Switch between LogReg and LinearSVC
- **Adjustable Threshold**: Configure confidence slider (0-100%)
- **Reference Images**: View 2-3 example images per species
- **Progress Bars**: Visual confidence indicators
- **Safety Guidelines**: Comprehensive do's and don'ts

### API Features

- **Batch Prediction**: Process multiple images
- **Probability Scores**: Full probability distribution
- **Safety Checking**: Automatic venomous detection
- **Flexible Loading**: Environment-based configuration

## 📖 Species Supported

Current dataset includes:
- Bungarus caeruleus (Common Krait) - Venomous
- Naja naja (Indian Cobra) - Venomous
- Ophiophagus hannah (King Cobra) - Venomous
- Hypnale hypnale (Hump-nosed Viper) - Venomous
- Craspedocephalus malabaricus (Malabar Pit Viper) - Venomous
- Craspedocephalus trigonocephalus (Sri Lankan Pit Viper) - Venomous
- Hypnale zara (Zara's Hump-nosed Viper) - Venomous
- Fowlea piscator (Checkered Keelback) - Non-venomous
- Ptyas mucosa (Indian Rat Snake) - Non-venomous
- Lycodon capucinus (Common Wolf Snake) - Non-venomous

## 🤝 Contributing

To add new species:
1. Add images to `data/snake_reference_images/<Species_name>/`
2. Retrain models: `python src/models/train.py`
3. Update `src/utils/safety.py` if venomous

## ⚠️ Disclaimer

**This system is for identification assistance only.**

- NOT a substitute for professional wildlife expertise
- NOT for medical diagnosis
- In case of snakebite: **seek immediate medical help**
- Always contact local wildlife or medical authorities
- Never handle snakes without proper training

## 📄 License

This project is for educational and research purposes.

## 👥 Authors

- Original project contributors
- Refactored and enhanced for production use

## 🙏 Acknowledgments

- Feature extraction using HOG, LBP, and HSV descriptors
- Scikit-learn for ML algorithms
- Streamlit for web framework
- PyTorch for feature computation

---

**Status**: ✅ Production-Ready
**Version**: 2.0 (Refactored)
**Last Updated**: March 28, 2026

🐍 Happy classifying! Always prioritize safety! ✨
