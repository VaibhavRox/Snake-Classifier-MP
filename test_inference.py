
import os
import random
import sys
from src.inference import SnakeClassifier

# Path to a random image from the dataset (from the 20 classes we processed)
# We know the classes are the first 20 alphabetically.
DATASET_PATH = "g:/Miniproject/Miniproject-Dataset"

def get_random_image():
    # List first 20 folders
    species_folders = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])[:20]
    
    # Pick a random species
    species = random.choice(species_folders)
    species_dir = os.path.join(DATASET_PATH, species)
    
    # Pick random image
    images = [f for f in os.listdir(species_dir) if f.lower().endswith(('.jpg', '.png'))]
    if not images:
        return None, None
    
    img_name = random.choice(images)
    return os.path.join(species_dir, img_name), species

def run_test():
    classifier = SnakeClassifier(model_path="src/models/RandomForest_model.pkl") # Use best model
    
    print("\n--- Running Inference Test ---\n")
    
    img_path, true_species = get_random_image()
    if not img_path:
        print("No images found.")
        return

    print(f"Testing on image: {img_path}")
    print(f"True Species: {true_species}\n")
    
    result = classifier.predict(img_path)
    
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print("Prediction Results:")
        for rank, res in enumerate(result["top_3"], 1):
            print(f"{rank}. {res['species']} (Prob: {res['probability']:.4f}) - {res['venomous_type']}")
            
        print(f"\nSafety Message: {result['safety_message']}")

if __name__ == "__main__":
    run_test()
