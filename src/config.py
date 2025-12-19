class_weights = {0:0.873, 1:0.873, 2:0.970, 3:0.970, 4:1.039, 5:1.453} 
SVM_CONFIG = {
    "kernel": "rbf",
    "C": 10.0,            
    "gamma": "scale",
    "probability": True, 
    "class_weight": class_weights
}
KNN_CONFIG = {
    "n_neighbors": 3,
    "weights": "distance",
    "metric": "euclidean"
}
# method: "flat", "hist_grad", "advanced", "efficientnet", "efficientnet_lbp", "resnet", "resnet_lbp"
SVM_FEATURE_EXTRACTION_METHOD = "efficientnet_lbp"
KNN_FEATURE_EXTRACTION_METHOD = "resnet_lbp"

UNKNOWN_THRESHOLD = 0.4 
TRASH_THRESHOLD = 0.35 
