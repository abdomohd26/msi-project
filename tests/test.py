import os
import cv2
import joblib
import numpy as np
from skimage.feature import local_binary_pattern


def _extract_lbp_features(image_bgr, P=8, R=1):
    """
    Extract LBP features (uniform patterns).
    Returns a fixed-length normalized histogram.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    lbp = local_binary_pattern(gray, P, R, method="uniform")

    hist, _ = np.histogram(
        lbp,
        bins=P + 2,
        range=(0, P + 2),
        density=True
    )

    return hist.astype(np.float32)


def predict(dataFilePath, bestModelPath):
    """
    Args:
        dataFilePath (str): Folder containing images directly.
        bestModelPath (str): Path to trained sklearn model (.pkl).

    Returns:
        list: Predictions for each image (sorted by filename).
    """

    # Load trained model
    model = joblib.load(bestModelPath)

    # Collect image files
    image_files = sorted([
        f for f in os.listdir(dataFilePath)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ])

    predictions = []

    for filename in image_files:
        image_path = os.path.join(dataFilePath, filename)

        image = cv2.imread(image_path)
        if image is None:
            continue  # Skip unreadable images safely

        # Resize (must match training!)
        image = cv2.resize(image, (128, 128))

        # Feature extraction
        features = _extract_lbp_features(image)

        # Model inference
        pred = model.predict([features])[0]

        predictions.append(pred)

    return predictions


# ===============================
# Local testing (safe for grading)
# ===============================
if __name__ == "__main__":
    preds = predict("tests/test_images", "models/svm/svm_model.pkl")
    print(preds)
