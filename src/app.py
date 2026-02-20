
import streamlit as st
import os
import numpy as np
from PIL import Image
from src.inference import SnakeClassifier

# Page Config
st.set_page_config(
    page_title="Snake Species Classifier",
    page_icon="🐍",
    layout="centered"
)

# Initialize Classifier (cached across sessions)
@st.cache_resource
def get_classifier():
    return SnakeClassifier()   # loads from ARTIFACTS_PATH in config

classifier = get_classifier()

# Title & Description
st.title("🐍 AI Snake Species Classifier")
st.markdown("""
This tool identifies snake species using **Classical Machine Learning**.
Upload an image to get the **Top-3 Predictions** and a **Safety Assessment**.
""")

# Sidebar
st.sidebar.header("About")
st.sidebar.info("""
**Model**: LinearSVC  
**Features**: HOG + LBP + HSV Colour  
**Pre-processing**: StandardScaler → PCA (300 components)  
**Classes**: 1,686 snake species
""")

# Image Upload
uploaded_file = st.file_uploader("Choose a snake image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Write to a temp file so OpenCV can read it
    tmp_path = "temp_image.jpg"
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.write("Analysing...")

    result = classifier.predict(tmp_path)

    # Clean up temp file
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    # Display Results
    if "error" in result:
        st.error(f"Error: {result['error']}")
    else:
        # Safety Banner
        safety_msg = result["safety_message"]
        if "UNKNOWN" in safety_msg:
            st.warning(f"⚠️ **{safety_msg}**")
            st.caption("The model rejects low-confidence predictions to prevent dangerous misidentification.")
        elif "DANGER" in safety_msg:
            st.error(f"☠️ **{safety_msg}**")
        else:
            st.success(f"✅ **{safety_msg}**")

        st.divider()

        # Top-3 Results
        st.subheader("Top Predictions")
        for i, res in enumerate(result["top_3"], 1):
            species  = res["species"]
            prob     = res["probability"]
            is_venom = "☠️ Venomous" if res["is_venomous"] else "✅ Non-venomous"

            with st.container():
                col1, col2 = st.columns([3, 1])
                col1.markdown(f"**{i}. {species}**")
                col2.markdown(f"`{prob * 100:.1f}%`")
                st.caption(f"{is_venom} ({res['venomous_type']})")
                st.progress(float(prob))

