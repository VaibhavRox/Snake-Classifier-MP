# 🚀 Quick Start Guide - Snake Classifier Streamlit App

## ✅ Setup Complete!

All tests have passed successfully. Your snake classification web app is ready to run!

---

## 📦 What's Included

### Main Application File
- **`app/streamlit_app.py`** - Complete Streamlit web application (365 lines)

### Key Features Implemented
✅ Image upload and display
✅ EfficientNet-B0 deep learning feature extraction (1280-dim)
✅ Logistic Regression & LinearSVC model support
✅ Top-3 predictions with confidence scores
✅ Unknown threshold logic (configurable, default 40%)
✅ Reference images dropdown viewer
✅ Venomous/Non-venomous detection
✅ Safety warnings and guidelines
✅ Professional UI with custom styling

---

## 🎯 Run the App

### Option 1: Default Settings
```bash
streamlit run app/streamlit_app.py
```

### Option 2: Custom Port
```bash
streamlit run app/streamlit_app.py --server.port 8080
```

### Option 3: Open Automatically in Browser
```bash
streamlit run app/streamlit_app.py --server.headless false
```

The app will be available at: **http://localhost:8501**

---

## 🎨 App Structure

### 1. Upload Section
- Drag & drop or browse for snake images
- Supports: JPG, JPEG, PNG formats
- Side-by-side display: uploaded image + results

### 2. Prediction Results
Shows:
- **Safety message**: Venomous/Non-venomous/Unknown
- **Confidence scores**: Color-coded progress bars
- **Top 3 predictions**: Scientific names with probabilities

Example output:
```
#1 Naja naja (Indian Cobra)
   Confidence: 62.5%
   Status: ☠️ Venomous
   Type: Naja

#2 Bungarus caeruleus (Common Krait)
   Confidence: 21.3%
   Status: ☠️ Venomous
   Type: Bungarus

#3 Ptyas mucosa (Rat Snake)
   Confidence: 10.2%
   Status: ✅ Non-venomous
   Type: Non-venomous (Likely)
```

### 3. Unknown Threshold Logic
If ALL predictions are below the threshold (default 40%):
```
⚠️ UNKNOWN SPECIES
Confidence too low for reliable identification.

Recommended Action:
- Keep a safe distance
- Contact local wildlife authorities
- Do not attempt to handle
- Seek medical help if bitten
```

### 4. Reference Images Dropdown
- Select any of the top-3 predicted species
- View 2-3 reference images from the training dataset
- Helps verify identification visually

### 5. Sidebar Controls
- **Model Selection**: Choose between LogReg (66.5% accuracy) or LinearSVC (62% accuracy)
- **Confidence Threshold**: Adjust from 0.0 to 1.0 (default 0.40)
- **Model Info**: See accuracy metrics and feature details
- **Instructions**: Quick how-to guide

---

## 📊 Performance Metrics

### Logistic Regression (Recommended)
- Top-1 Accuracy: **66.5%**
- Top-3 Accuracy: **90.5%**
- Top-5 Accuracy: **97.5%**
- Classes: **10 snake species**

### LinearSVC
- Top-1 Accuracy: **62.0%**
- Top-3 Accuracy: **83.0%**
- Top-5 Accuracy: **91.0%**
- Classes: **10 snake species**

---

## 🔧 Customization Options

### Adjust Confidence Threshold
In the sidebar, move the slider to change the minimum confidence:
- **0.20-0.30**: More permissive (may show low-confidence predictions)
- **0.40** (default): Balanced approach
- **0.50-0.70**: Strict (only high-confidence predictions)

### Switch Models
Use the dropdown to select:
- **logreg** (default, better accuracy)
- **linearsvc** (alternative model)

### Modify Reference Images
Add more species:
1. Create a folder in `data/snake_reference_images/` with the scientific name
2. Add 2-3 reference images (JPG/PNG)
3. Images will automatically appear in the dropdown

---

## 🔬 Technical Details

### ML Pipeline
1. **Input**: User uploads image (any size)
2. **Preprocessing**: Resize to 224×224, ImageNet normalization
3. **Feature Extraction**: EfficientNet-B0 → 1280-dim feature vector
4. **Scaling**: StandardScaler (fitted on training data)
5. **Classification**: Logistic Regression (OvR multi-class)
6. **Output**: Top-3 species with probabilities

### Model Files Used
```
src/models/artifacts/logreg/
├── model.pkl         # Logistic Regression classifier (50.95 KB)
├── scaler.pkl        # StandardScaler (30.60 KB)
└── label_names.pkl   # Species names (1.47 KB)
```

### Functions Implemented

**Model Loading**
- `load_model(model_type)` - Cached classifier loading

**Feature Extraction**
- `extract_features(image_path)` - EfficientNet-B0 feature extraction
- From `src/features/extractors.py`

**Prediction**
- `predict_snake(classifier, image_path)` - Complete inference pipeline
- Returns: top_3 predictions + safety_message

**Top-3 Processing**
- `get_top3_predictions(result)` - Extracts top-3 species
- Sorted by probability (descending)

**Threshold Checking**
- `check_unknown_threshold(predictions, threshold)` - Safety logic
- Returns: True if max_prob < threshold

**Reference Images**
- `get_reference_images(species_name, max_images)` - Cached image retrieval
- `show_reference_images(species_name)` - UI display

---

## 📝 Testing

Run the test suite to verify everything works:
```bash
python test_setup.py
```

Expected output:
```
[OK] PASS: Imports
[OK] PASS: Project Structure
[OK] PASS: Model Artifacts
[OK] PASS: Reference Images
[OK] PASS: Classifier Loading
[OK] PASS: Feature Extraction

Results: 6/6 tests passed
```

---

## 🐛 Common Issues & Solutions

### "Model not loaded" error
**Cause**: Missing model artifacts
**Solution**: Ensure `src/models/artifacts/logreg/` contains all `.pkl` files

### No reference images shown
**Cause**: Species name mismatch
**Solution**: Folder name must exactly match the scientific name (e.g., `Naja_naja`)

### Slow first prediction
**Cause**: EfficientNet-B0 model downloads on first use (20.5 MB)
**Solution**: Normal behavior, subsequent predictions will be fast

### Sklearn version warning
**Cause**: Model trained with sklearn 1.6.1, running with 1.7.2
**Solution**: Warning can be ignored or retrain model with current sklearn version

---

## 🌟 Usage Example

```bash
# 1. Activate virtual environment
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# 2. Run the app
streamlit run app/streamlit_app.py

# 3. Open browser to http://localhost:8501

# 4. Upload a snake image

# 5. View results and reference images
```

---

## 📚 Additional Files

- **`STREAMLIT_README.md`** - Comprehensive documentation
- **`test_setup.py`** - System verification script
- **`requirements.txt`** - Python dependencies (updated with torch, Pillow)

---

## ⚠️ Important Disclaimers

**Medical Safety**
- This system is for **identification assistance only**
- NOT a substitute for professional medical advice
- In case of snakebite: **seek immediate medical help**
- Contact local wildlife or medical authorities

**Prediction Accuracy**
- System achieves 66.5% top-1 accuracy, 90.5% top-3 accuracy
- Misidentification is possible
- Use as a guide, not definitive identification
- When in doubt, treat as potentially venomous

---

## 🎉 You're All Set!

Your snake classification web app is ready. Run it now:

```bash
streamlit run app/streamlit_app.py
```

Happy classifying! 🐍✨

---

**Questions?** Check `STREAMLIT_README.md` for detailed documentation.
