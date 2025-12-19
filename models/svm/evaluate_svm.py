import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.trash_unknown_handler import handle_trash_unknown
from src.config import SVM_FEATURE_EXTRACTION_METHOD

model = joblib.load("models/svm/svm_model.pkl")
X_val = np.load(Path(f"features/{SVM_FEATURE_EXTRACTION_METHOD}/val_features.npy").resolve())
y_val = np.load(Path(f"features/{SVM_FEATURE_EXTRACTION_METHOD}/val_labels.npy").resolve())

y_pred = np.array([handle_trash_unknown(model, x) for x in X_val])

classes_present = np.unique(np.concatenate([y_val, y_pred]))

print("Accuracy:", accuracy_score(y_val, y_pred))
print("\nClassification Report:\n", classification_report(y_val, y_pred, labels=classes_present, zero_division=0))
conf =  confusion_matrix(y_val, y_pred, labels=classes_present)

plt.figure(figsize=(10, 8))
sns.heatmap(conf, annot=True, fmt='d', cmap='Blues', xticklabels=classes_present, yticklabels=classes_present)
plt.title('Confusion Matrix For SVM')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('models/svm/confusion_matrix_svm.png')
plt.show()
