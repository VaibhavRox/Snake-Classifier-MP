# 🐍 AI Snake Species Classifier

This project is an AI-powered tool designed to classify snake species from images using **Classical Machine Learning** techniques. It identifies top predictions and provides crucial safety assessments (venomous vs. non-venomous).

## 🚀 Features

- **Multi-Feature Extraction**: Utilizes Histogram of Oriented Gradients (HOG), Local Binary Patterns (LBP), and HSV color histograms for robust image analysis.
- **Classical ML Models**: capable of using Random Forest, Linear SVM, and KNN classifiers.
- **Safety First**: Includes a "Safety Gate" system that flags low-confidence predictions as "UNKNOWN" and clearly warns users about venomous species.
- **Interactive UI**: Built with Streamlit for easy image uploading and real-time inference.

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/VaibhavRox/Miniproject-AI-based-Snake-Classification.git
    cd Miniproject-AI-based-Snake-Classification
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Prepare the Dataset (if training):**
    *   Ensure your dataset is placed in `Miniproject-Dataset/`.
    *   The code expects a structure compatible with the training scripts in `src/models/train.py`.

## 🏃 Usage

### Running the Web App
To start the interactive web interface:

```bash
streamlit run src/app.py
```

Upload an image of a snake to see the predicted species, probability scores, and safety alert.

### Inference Script
You can also run inference programmatically using `src/inference.py` or test it via:

```bash
python test_inference.py
```

## 📂 Project Structure

```
.
├── Miniproject-Dataset/    # Dataset images (ignored by git)
├── data/                   # Processed data
├── src/
│   ├── features/           # Feature extraction logic (extractors.py, pipeline.py)
│   ├── models/             # Trained models (.pkl) & training scripts
│   ├── utils/              # Config and safety utilities
│   ├── app.py              # Streamlit application entry point
│   └── inference.py        # Core inference class
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## ⚠️ Disclaimer

This tool is for educational and experimental purposes. **Do not rely solely on this AI for identification in dangerous situations.** detailed herpetological knowledge should always verify the results.
