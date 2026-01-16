
import numpy as np

# Keyword-based venomous detection for mini-project
# In a real system, this would be a comprehensive database or CSV lookup.
VENOMOUS_KEYWORDS = [
    "vipera", "crotalus", "sistrurus", "agkistrodon", "bitis", "cerastes", "echis", 
    "macrovipera", "pseudocerastes", "daboia", "causus", "naja", "bungarus", 
    "dendroaspis", "micrurus", "ophiophagus", "elapsoidea", "walterinnesia",
    "hydrophis", "laticauda", "atractaspis", "trimeresurus", "tropidolaemus", 
    "protobothrops", "gloydius", "ovophis", "bothrops", "lachesis", "porthidium", 
    "bothriechis", "atheris", "calloselasma", "azemiops", "deinagkistrodon"
]

def is_venomous(species_name):
    """
    Determines if a snake species is venomous based on scientific name keywords.
    """
    name_lower = species_name.lower()
    for kw in VENOMOUS_KEYWORDS:
        if kw in name_lower:
            return True, kw.capitalize() # Return True and the matched genus/keyword
    return False, "Non-venomous (Likely)"

def check_safety(predictions, label_names, threshold=0.6):
    """
    Applies safety threshold logic.
    predictions: Probability estimates for one sample (from model.predict_proba).
    label_names: Array of class names.
    threshold: Confidence threshold.
    
    Returns:
    - top_3: List of (species, probability, is_venomous_bool)
    - safety_message: String ("SAFE", "WARNING: Venomous", "UNKNOWN")
    """
    # Sort indices by probability descending
    top_indices = np.argsort(predictions)[::-1][:3]
    top_probs = predictions[top_indices]
    
    top_3_results = []
    
    max_conf = top_probs[0]
    
    for idx, prob in zip(top_indices, top_probs):
        species = label_names[idx]
        venomous, keyword = is_venomous(species)
        top_3_results.append({
            "species": species,
            "probability": prob,
            "is_venomous": venomous,
            "venomous_type": keyword
        })
        
    if max_conf < threshold:
        return top_3_results, "UNKNOWN – Unable to identify safely"
    
    # Check if top result is venomous
    if top_3_results[0]["is_venomous"]:
        return top_3_results, "DANGER: High probability of venomous species!"
        
    return top_3_results, "Likely Non-venomous"
