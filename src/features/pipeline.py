
import os
import numpy as np
import cv2
from tqdm import tqdm
from src.utils.config import DATASET_PATH, PROCESSED_DATA_PATH
from src.features.extractors import extract_all_features

def process_dataset(max_images_per_class=20, max_classes=20):
    """
    Iterates through the dataset, extracts features, and saves them.
    max_images_per_class: Limit to save time during development.
    max_classes: Limit number of species to process for demo.
    """
    if not os.path.exists(PROCESSED_DATA_PATH):
        os.makedirs(PROCESSED_DATA_PATH)

    features = []
    labels = []
    label_names = []
    
    # Get list of species folders
    species_folders = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])
    
    if max_classes:
        species_folders = species_folders[:max_classes]

    print(f"Found {len(species_folders)} species classes (limited to {max_classes}).")
    
    for label_idx, species in enumerate(tqdm(species_folders, desc="Processing Species")):
        species_dir = os.path.join(DATASET_PATH, species)
        label_names.append(species)
        
        image_files = [f for f in os.listdir(species_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Limit images per class
        image_files = image_files[:max_images_per_class]
        
        for img_file in image_files:
            img_path = os.path.join(species_dir, img_file)
            
            # Extract Features
            feat_vector = extract_all_features(img_path)
            
            if feat_vector is not None:
                features.append(feat_vector)
                labels.append(label_idx)
    
    # Convert to numpy arrays
    X = np.array(features)
    y = np.array(labels)
    
    print(f"Extraction Complete. Feature Matrix Shape: {X.shape}")
    print(f"Labels Shape: {y.shape}")
    
    # Save Data
    np.save(os.path.join(PROCESSED_DATA_PATH, "features.npy"), X)
    np.save(os.path.join(PROCESSED_DATA_PATH, "labels.npy"), y)
    np.save(os.path.join(PROCESSED_DATA_PATH, "label_names.npy"), np.array(label_names))
    
    print(f"Data saved to {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    # You can adjust max_images_per_class here
    process_dataset(max_images_per_class=10, max_classes=20) 
