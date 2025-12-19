import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
from src.config import SVM_CONFIG, SVM_FEATURE_EXTRACTION_METHOD
from pathlib import Path

abs_path = Path(f"features/{SVM_FEATURE_EXTRACTION_METHOD}/train_features.npy").resolve()
abs_path1 = Path(f"features/{SVM_FEATURE_EXTRACTION_METHOD}/train_labels.npy").resolve()

X_train = np.load(abs_path)
y_train = np.load(abs_path1)

svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(**SVM_CONFIG))
])

print("Training SVM...")
svm_pipeline.fit(X_train, y_train)

joblib.dump(svm_pipeline, "models/svm/svm_model.pkl")
print("SVM model saved.")
