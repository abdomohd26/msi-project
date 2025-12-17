import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# Load model
model = joblib.load("models/svm/svm_model.pkl")

# Load validation data
x_val = np.load("../features/val_features.npy")
y_val = np.load("../features/val_labels.npy")

# Predict
y_pred = model.predict(X_val)

print("Accuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:\n", classification_report(y_val, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_pred))
