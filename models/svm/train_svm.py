import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
from src.config import SVM_CONFIG
from src.utils import load_dataset

# Load data
X_train = np.load("../features/train_features.npy")
y_train = np.load("../features/train_labels.npy")

# Pipeline = Scaling + SVM
svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(**SVM_CONFIG))
])

print("Training SVM...")
svm_pipeline.fit(X_train, y_train)

# Save model
joblib.dump(svm_pipeline, "models/svm/svm_model.pkl")
print("SVM model saved.")
