
import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
from src.inference import SnakeClassifier

# Page Config
st.set_page_config(
    page_title="Snake Species Classifier",
    page_icon="🐍",
    layout="centered"
)

# Initialize Classifier
@st.cache_resource
def get_classifier():
    return SnakeClassifier(model_path="src/models/RandomForest_model.pkl")

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
**Model**: Random Forest
**Features**: HOG + LBP + HSV Color
**Classes**: 20 (Demo Subset)
""")

# Image Upload
uploaded_file = st.file_uploader("Choose a snake image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display Image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Save temp file for OpenCV processing
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.write("Analyzing...")
    
    # Predict
    result = classifier.predict("temp_image.jpg")
    
    # Remove temp file
    if os.path.exists("temp_image.jpg"):
        os.remove("temp_image.jpg")
    
    # Display Results
    if "error" in result:
        st.error(f"Error: {result['error']}")
    else:
        # Safety Banner
        safety_msg = result['safety_message']
        if "UNKNOWN" in safety_msg:
            st.warning(f"⚠️ **{safety_msg}**")
            st.caption("The model strictly rejects low-confidence predictions to prevent dangerous errors.")
        elif "DANGER" in safety_msg:
            st.error(f"☠️ **{safety_msg}**")
        else:
            st.success(f"✅ **{safety_msg}**")
            
        st.divider()
        
        # Top 3 Results
        st.subheader("Top Predictions")
        for i, res in enumerate(result['top_3'], 1):
            species = res['species']
            prob = res['probability']
            is_venom = "☠️ Venomous" if res['is_venomous'] else "✅ Non-venomous"
            
            with st.container():
                col1, col2 = st.columns([3, 1])
                col1.markdown(f"**{i}. {species}**")
                col2.markdown(f"`{prob*100:.1f}%`")
                st.caption(f"{is_venom} ({res['venomous_type']})")
                st.progress(float(prob))
