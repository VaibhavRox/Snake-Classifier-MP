
import os
import random
from src.inference import SnakeClassifier
from src.utils.config import DATASET_PATH


def get_random_image(dataset_path=DATASET_PATH):
    """Pick a random image from a random species folder in the dataset."""
    if not os.path.exists(dataset_path):
        print(f"Dataset path not found: {dataset_path}")
        print("Set the SNAKE_DATASET_PATH environment variable to your dataset location.")
        return None, None

    species_folders = sorted([
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ])

    if not species_folders:
        print("No species folders found in dataset path.")
        return None, None

    species     = random.choice(species_folders)
    species_dir = os.path.join(dataset_path, species)

    images = [
        f for f in os.listdir(species_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    if not images:
        return None, None

    img_name = random.choice(images)
    return os.path.join(species_dir, img_name), species


def run_test():
    # Load the classifier (uses ARTIFACTS_PATH from config)
    classifier = SnakeClassifier()

    print("\n--- Running Inference Test ---\n")

    img_path, true_species = get_random_image()
    if not img_path:
        print("Could not find a test image. Exiting.")
        return

    print(f"Image      : {img_path}")
    print(f"True class : {true_species}\n")

    result = classifier.predict(img_path)

    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print("Top-3 Predictions:")
        for rank, res in enumerate(result["top_3"], 1):
            print(
                f"  {rank}. {res['species']:<40}  "
                f"prob={res['probability']:.4f}  "
                f"{'☠ VENOMOUS' if res['is_venomous'] else '✓ non-venomous'}"
            )
        print(f"\nSafety message: {result['safety_message']}")


if __name__ == "__main__":
    run_test()

