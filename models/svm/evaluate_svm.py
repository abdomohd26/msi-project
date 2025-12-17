import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from pathlib import Path
# Load model
model = joblib.load("models/svm/svm_model.pkl")

abs_path = Path("features/val_features.npy").resolve()
abs_path1 = Path("features/val_labels.npy").resolve()

# Load validation data
X_val = np.load(abs_path)
y_val = np.load(abs_path1)

# Predict
y_pred = model.predict(X_val)

print("Accuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:\n", classification_report(y_val, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_pred))
