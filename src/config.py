class_weights = {0:1.0, 1:1.0, 2:1.2, 3:1.5, 4:1.5, 5:2.0} 
SVM_CONFIG = {
    "kernel": "rbf",       # linear / rbf / poly
    "C": 10.0,            # regularization parameter
    "gamma": "scale",
    "probability": True, 
    "class_weight": class_weights
}

UNKNOWN_THRESHOLD = 0.6   # confidence threshold
