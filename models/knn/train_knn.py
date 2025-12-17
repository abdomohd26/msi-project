import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pathlib import Path
from src.config import KNN_CONFIG

abs_path = Path("features/train_features.npy").resolve()
abs_path1 = Path("features/train_labels.npy").resolve()

# Load data
X_train = np.load(abs_path)
y_train = np.load(abs_path1)

# Pipeline = Scaling + KNN
knn_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(**KNN_CONFIG))
])

print("Training KNN...")
knn_pipeline.fit(X_train, y_train)

# Save model
joblib.dump(knn_pipeline, "models/knn/knn_model.pkl")
print("KNN model saved.")