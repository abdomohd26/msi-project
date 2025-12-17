from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from pathlib import Path
import joblib
from src.trash_unknown_handler import handle_trash_unknown

TRASH_CLASS = 5
UNKNOWN_CLASS = 6

# Load model and validation features
model = joblib.load("models/svm/svm_model.pkl")
X_val = np.load("features/val_features.npy")
y_val = np.load("features/val_labels.npy")

# Predict with Trash + Unknown handling
y_pred = np.array([handle_trash_unknown(model, x, trash_class=TRASH_CLASS) for x in X_val])

# Identify classes present in y_val + possible Unknown predictions
classes_present = np.unique(np.concatenate([y_val, y_pred]))

# Classification report only for present classes
print("Accuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:\n", classification_report(y_val, y_pred, labels=classes_present, zero_division=0))
print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_pred, labels=classes_present))
