import os
import cv2
import joblib
import json
import numpy as np
import sys
from src.unknown_handler import handle_unknown
from features.cnn_feature_extraction import image_to_feature_efficientnet_lbp
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

MODEL_PATH = os.path.join("models", "svm", "svm_model.pkl")
MAPPING_PATH = os.path.join("deployment", "class_mapping.json")
PREPROCESS_SIZE = (128, 128)

_model = None
_class_mapping = None

def load_resources():
    """
    loading model and class mapping.
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
            _class_mapping = {int(k): v for k, v in data.items()}

def predict(image_bgr):
    """
    Run inference on a single BGR image (numpy array).
    
    Args:
        image_bgr (np.ndarray): Input image in BGR format (from OpenCV).
        
    Returns:
        tuple: (class_id, label_str, confidence_score)
    """
    load_resources()
    
    if image_bgr is None:
        raise ValueError("Input image is None")


    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    image_resized = cv2.resize(image_rgb, PREPROCESS_SIZE)
    
    img_arr = image_resized.astype(np.float32) / 255.0
    
    features = image_to_feature_efficientnet_lbp(img_arr)

    final_class_id = handle_unknown(_model, features)
    
    probs = _model.predict_proba([features])[0]
    confidence = float(np.max(probs))
    
    if final_class_id == 6:
        label = _class_mapping.get(6, "Unknown")
    else:
        label = _class_mapping.get(final_class_id, "Unknown")
        
    return final_class_id, label, confidence
