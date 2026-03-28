"""
Snake Species Identification and Safety Assistance
-------------------------------------------------
A Streamlit web application for snake classification using deep learning
feature extraction (EfficientNet-B0) and Logistic Regression.
"""

import os
import streamlit as st
import numpy as np
from PIL import Image
import glob
from pathlib import Path

# Import the classifier and utilities
from src.inference.predictor import SnakeClassifier
from src.utils.config import PROJECT_ROOT


# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Snake Species Identification",
    page_icon="🐍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS STYLING
# ============================================================
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        padding: 1rem 0;
    }
    .warning-box {
        background-color: #FFF3CD;
        border-left: 5px solid #FFC107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .prediction-box {
        background-color: #F5F5F5;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================
# 1. LOAD MODEL (CACHED)
# ============================================================
@st.cache_resource
def load_model(model_type="logreg"):
    """
    Load the trained snake classifier model.

    Parameters
    ----------
    model_type : str
        Type of model to load ("logreg" or "linearsvc")

    Returns
    -------
    SnakeClassifier
        Loaded classifier instance
    """
    return SnakeClassifier(model_type=model_type)


# ============================================================
# 2. LOAD REFERENCE IMAGES
# ============================================================
@st.cache_data
def get_reference_images(species_name, max_images=3):
    """
    Get reference images for a given species.

    Parameters
    ----------
    species_name : str
        Scientific name of the species
    max_images : int
        Maximum number of reference images to return

    Returns
    -------
    list
        List of image file paths
    """
    # Path to reference images (using the data/snake_reference_images folder)
    reference_dir = os.path.join(PROJECT_ROOT, "data", "snake_reference_images", species_name)

    if not os.path.exists(reference_dir):
        return []

    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []

    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(reference_dir, ext)))

    # Return up to max_images
    return sorted(image_files)[:max_images]


# ============================================================
# 3. PREDICTION FUNCTION
# ============================================================
def predict_snake(classifier, image_path):
    """
    Run prediction on uploaded image.

    Parameters
    ----------
    classifier : SnakeClassifier
        The loaded classifier
    image_path : str
        Path to the image

    Returns
    -------
    dict
        Prediction results containing top_3 and safety_message
    """
    return classifier.predict(image_path)


# ============================================================
# 4. GET TOP-3 PREDICTIONS
# ============================================================
def get_top3_predictions(result):
    """
    Extract and format top-3 predictions from result.

    Parameters
    ----------
    result : dict
        Prediction result from classifier

    Returns
    -------
    list
        List of top-3 predictions with species, probability, and venomous info
    """
    if "top_3" in result:
        return result["top_3"]
    return []


# ============================================================
# 5. CHECK UNKNOWN THRESHOLD
# ============================================================
def check_unknown_threshold(top_predictions, threshold=0.40):
    """
    Check if prediction confidence is below threshold.

    Parameters
    ----------
    top_predictions : list
        List of top predictions
    threshold : float
        Confidence threshold (default: 0.40)

    Returns
    -------
    bool
        True if all predictions are below threshold (unknown)
    """
    if not top_predictions:
        return True

    # Check if top prediction is below threshold
    max_prob = top_predictions[0]["probability"]
    return max_prob < threshold


# ============================================================
# 6. SHOW REFERENCE IMAGES
# ============================================================
def show_reference_images(species_name):
    """
    Display reference images for a selected species.

    Parameters
    ----------
    species_name : str
        Scientific name of the species
    """
    ref_images = get_reference_images(species_name, max_images=3)

    if ref_images:
        st.markdown(f"### Reference Images: *{species_name.replace('_', ' ')}*")

        cols = st.columns(min(len(ref_images), 3))
        for idx, img_path in enumerate(ref_images):
            with cols[idx]:
                try:
                    img = Image.open(img_path)
                    st.image(img, use_container_width=True, caption=f"Reference {idx+1}")
                except Exception as e:
                    st.error(f"Error loading image: {e}")
    else:
        st.warning(f"No reference images available for {species_name.replace('_', ' ')}")


# ============================================================
# MAIN APP
# ============================================================
def main():
    # Title
    st.markdown('<div class="main-title">🐍 Snake Species Identification and Safety Assistance</div>',
                unsafe_allow_html=True)

    # Warning Banner
    st.markdown("""
        <div class="warning-box">
            <strong>⚠️ IMPORTANT DISCLAIMER:</strong><br>
            This system is for <strong>identification assistance only</strong>.
            In case of snakebite, seek medical help immediately. Do not rely solely on this
            identification for medical decisions. Always contact local wildlife or medical authorities.
        </div>
        """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # Model selection
        model_type = st.selectbox(
            "Select Model",
            options=["logreg", "linearsvc"],
            index=0,
            help="LogReg generally performs better with 66.5% accuracy"
        )

        # Confidence threshold
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.40,
            step=0.05,
            help="Minimum confidence required for prediction"
        )

        st.markdown("---")

        # About section
        st.header("📊 Model Info")
        st.info(f"""
        **Model Type:** {model_type.upper()}
        **Features:** EfficientNet-B0 (1280-dim)
        **Accuracy:** {'66.5%' if model_type == 'logreg' else '62.0%'}
        **Top-3 Accuracy:** {'90.5%' if model_type == 'logreg' else '83.0%'}
        **Classes:** 10 snake species
        """)

        st.markdown("---")

        # Instructions
        st.header("📖 How to Use")
        st.markdown("""
        1. Upload a snake image
        2. View prediction results
        3. Check confidence scores
        4. View reference images
        5. Take appropriate safety measures
        """)

    # Load classifier
    with st.spinner("Loading model..."):
        classifier = load_model(model_type)

    # Main content area
    st.markdown("---")

    # File uploader
    uploaded_file = st.file_uploader(
        "📤 Upload Snake Image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )

    if uploaded_file is not None:
        # Create two columns
        col1, col2 = st.columns([1, 1])

        # Display uploaded image
        with col1:
            st.markdown("### Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)

        # Save temporary file for prediction
        temp_path = "temp_upload.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Run prediction
        with st.spinner("Analyzing image..."):
            result = predict_snake(classifier, temp_path)

        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

        # Display results
        with col2:
            st.markdown("### Analysis Results")

            if "error" in result:
                st.error(f"❌ Error: {result['error']}")
            else:
                top_3 = get_top3_predictions(result)
                is_unknown = check_unknown_threshold(top_3, confidence_threshold)

                # Unknown threshold check
                if is_unknown:
                    st.error("""
                    ### ⚠️ UNKNOWN SPECIES

                    **Confidence too low for reliable identification.**

                    **Recommended Action:**
                    - Keep a safe distance
                    - Contact local wildlife authorities
                    - Do not attempt to handle
                    - Seek medical help if bitten
                    """)
                else:
                    # Display safety message
                    safety_msg = result.get("safety_message", "")

                    if "DANGER" in safety_msg:
                        st.error(f"☠️ **{safety_msg}**")
                    elif "UNKNOWN" in safety_msg:
                        st.warning(f"⚠️ **{safety_msg}**")
                    else:
                        st.success(f"✅ **{safety_msg}**")

        # Top-3 Predictions Section
        st.markdown("---")
        st.markdown("## 🎯 Top 3 Predictions")

        if not is_unknown and top_3:
            # Display predictions in columns
            pred_cols = st.columns(3)

            for idx, pred in enumerate(top_3):
                with pred_cols[idx]:
                    species = pred["species"]
                    prob = pred["probability"]
                    is_venom = pred["is_venomous"]
                    venom_type = pred["venomous_type"]

                    # Format species name for display
                    display_name = species.replace("_", " ")

                    # Create prediction card
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h3 style="color: {'#D32F2F' if is_venom else '#388E3C'};">
                            #{idx+1} {display_name}
                        </h3>
                        <p><strong>Confidence:</strong> {prob*100:.1f}%</p>
                        <p><strong>Status:</strong> {'☠️ Venomous' if is_venom else '✅ Non-venomous'}</p>
                        <p><strong>Type:</strong> {venom_type}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Progress bar for confidence
                    st.progress(float(prob))

        # Reference Images Section
        if top_3 and not is_unknown:
            st.markdown("---")
            st.markdown("## 📸 Reference Images")

            # Create dropdown with species names
            species_options = [pred["species"] for pred in top_3]
            display_options = [s.replace("_", " ") for s in species_options]

            selected_display = st.selectbox(
                "Select a species to view reference images:",
                options=display_options,
                index=0
            )

            # Get the actual species name
            selected_idx = display_options.index(selected_display)
            selected_species = species_options[selected_idx]

            # Show reference images
            show_reference_images(selected_species)

        # Additional Safety Information
        st.markdown("---")
        st.markdown("## 🚨 Safety Guidelines")

        safety_col1, safety_col2 = st.columns(2)

        with safety_col1:
            st.markdown("""
            ### If the snake is venomous:
            - **Maintain safe distance** (at least 6 feet)
            - **Do not attempt to catch** or handle
            - **Call wildlife control** immediately
            - **Note the time and location** if bitten
            - **Seek immediate medical attention** if bitten
            """)

        with safety_col2:
            st.markdown("""
            ### General precautions:
            - **Never provoke** any snake
            - **Keep pets and children** away
            - **Document with photos** from safe distance
            - **Report to local authorities** if near populated areas
            - **Remember:** Even non-venomous bites can cause infection
            """)

    else:
        # Instructions when no file is uploaded
        st.info("""
        ### 👆 Please upload an image to begin

        **Tips for best results:**
        - Use clear, well-lit images
        - Ensure the snake is clearly visible
        - Avoid blurry or low-resolution images
        - Multiple angles can help with identification
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; padding: 1rem;">
            <p><strong>Snake Species Classifier</strong> | Powered by EfficientNet-B0 & Logistic Regression</p>
            <p>For educational and assistance purposes only. Not a substitute for professional wildlife expertise.</p>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# RUN APP
# ============================================================
if __name__ == "__main__":
    main()
