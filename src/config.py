SVM_CONFIG = {
    "kernel": "rbf",       # linear / rbf / poly
    "C": 10.0,            # regularization parameter
    "gamma": "scale",
    "probability": True, 
    "class_weight": "balanced"
}

UNKNOWN_THRESHOLD = 0.6   # confidence threshold
