import cv2
import joblib
import json
import numpy as np
import os

# Import project modules
# Assumes running from project root or src is in pythonpath
from src.unknown_handler import handle_unknown
from features.cnn_feature_extraction import image_to_feature_cnn

# Configuration
MODEL_PATH = os.path.join("models", "svm", "svm_model.pkl")
MAPPING_PATH = os.path.join("deployment", "class_mapping.json")
PREPROCESS_SIZE = (128, 128)

# Global resources
_model = None
_class_mapping = None

def _load_resources():
    """
    Lazy load model and class mapping.
    """
    global _model, _class_mapping
    
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please run training first.")
        print(f"Loading model from {MODEL_PATH}...")
        _model = joblib.load(MODEL_PATH)
        
    if _class_mapping is None:
        if not os.path.exists(MAPPING_PATH):
            raise FileNotFoundError(f"Mapping file not found at {MAPPING_PATH}.")
        with open(MAPPING_PATH, 'r') as f:
            data = json.load(f)
            # Ensure keys are integers
            _class_mapping = {int(k): v for k, v in data.items()}

def predict(image_bgr):
    """
    Run inference on a single BGR image (numpy array).
    
    Args:
        image_bgr (np.ndarray): Input image in BGR format (from OpenCV).
        
    Returns:
        tuple: (class_id, label_str, confidence_score)
    """
    _load_resources()
    
    if image_bgr is None:
        raise ValueError("Input image is None")

    # 1. Preprocessing
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Resize to match training data generation (128x128)
    # The CNN extractor will handle further resizing/normalization if needed (internally resizes to 224)
    image_resized = cv2.resize(image_rgb, PREPROCESS_SIZE)
    
    # Normalize to [0, 1] as expected by image_to_feature_cnn logic adaptation
    # (The original image_to_array function does this /255.0)
    img_arr = image_resized.astype(np.float32) / 255.0
    
    # 2. Feature Extraction
    # returns ID feature vector (e.g. 512 dims for RestNet18)
    features = image_to_feature_cnn(img_arr)
    
    # 3. Prediction & Unknown Handling
    # handle_unknown checks confidence threshold and returns 6 if below threshold
    final_class_id = handle_unknown(_model, features)
    
    # 4. Get raw confidence for display
    # We calculate this again to return it
    probs = _model.predict_proba([features])[0]
    confidence = float(np.max(probs))
    
    # 5. Map label
    if final_class_id == 6: # Unknown class ID based on prompt/handler
        label = _class_mapping.get(6, "Unknown")
        # If rejected, the confidence of the top class was low, 
        # but we return that low confidence value as the score.
    else:
        label = _class_mapping.get(final_class_id, "Unknown")
        
    return final_class_id, label, confidence
