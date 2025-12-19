import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pathlib import Path
from src.config import KNN_CONFIG, KNN_FEATURE_EXTRACTION_METHOD

abs_path = Path(f"features/{KNN_FEATURE_EXTRACTION_METHOD}/train_features.npy").resolve()
abs_path1 = Path(f"features/{KNN_FEATURE_EXTRACTION_METHOD}/train_labels.npy").resolve()

X_train = np.load(abs_path)
y_train = np.load(abs_path1)

knn_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(**KNN_CONFIG))
])

print("Training KNN...")
knn_pipeline.fit(X_train, y_train)

joblib.dump(knn_pipeline, "models/knn/knn_model.pkl")
print("KNN model saved.")
