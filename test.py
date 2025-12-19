import os
import cv2
import joblib
import json
from features.cnn_feature_extraction import image_to_feature_efficientnet_lbp



def predict(dataFilePath, bestModelPath):
    model = joblib.load(bestModelPath)

    with open("deployment/class_mapping.json", 'r') as f:
        class_mapping = json.load(f)
    class_mapping = {int(k): v for k, v in class_mapping.items()}

    image_files = sorted([
        f for f in os.listdir(dataFilePath)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ])

    predictions = []

    for filename in image_files:
        image_path = os.path.join(dataFilePath, filename)

        image = cv2.imread(image_path)
        if image is None:
            continue
        
        image = cv2.resize(image, (128, 128))
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        features = image_to_feature_efficientnet_lbp(img_rgb)
        pred = model.predict([features])[0]
        predictions.append(class_mapping.get(pred, "Unknown"))

    return predictions 


if __name__ == "__main__":
    data_path = "tests/test_images"
    preds = predict(data_path, "models/svm/svm_model.pkl")
    print(preds)
