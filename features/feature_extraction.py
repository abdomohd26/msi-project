import csv
from pathlib import Path

import numpy as np
from PIL import Image
from .hand_crafted_feature_extraction import (
    image_to_feature_flat,
    image_to_feature_hist_grad,
    image_to_feature_advanced,
)
from .cnn_feature_extraction import (
    image_to_feature_efficientnet,
    image_to_feature_efficientnet_lbp,
    image_to_feature_resnet,
    image_to_feature_resnet_lbp,
)
from src.config import KNN_FEATURE_EXTRACTION_METHOD, SVM_FEATURE_EXTRACTION_METHOD


def load_split_csv(csv_path):
    """
    Read CSV with columns: path,label -> returns lists (paths, labels).
    """
    paths = []
    labels = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # skip header if present
        for row in reader:
            if len(row) < 2:
                continue
            paths.append(row[0])
            labels.append(int(row[1]))
    return paths, np.array(labels, dtype=np.int64)

def image_to_array(path, size=(64, 64)):
    """
    Open image, convert to RGB, resize, return as numpy array [H, W, 3] in [0,1].
    """
    img = Image.open(path).convert("RGB")
    img = img.resize(size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def paths_to_features(csv_path, output_prefix, method="hist_grad", size=(64, 64), bins=16):
    """
    """
    paths, labels = load_split_csv(csv_path)

    features = []
    for i, p in enumerate(paths):
        try:
            arr = image_to_array(p, size=size)
            if method == "flat":
                feat = image_to_feature_flat(arr)
            elif method == "hist_grad":
                feat = image_to_feature_hist_grad(arr, bins=bins)
            elif method == "advanced":
                feat = image_to_feature_advanced(arr)
            elif method == "efficientnet":
                feat = image_to_feature_efficientnet(arr)
            elif method == "efficientnet_lbp":
                feat = image_to_feature_efficientnet_lbp(arr)
            elif method == "resnet":
                feat = image_to_feature_resnet(arr)
            elif method == "resnet_lbp":
                feat = image_to_feature_resnet_lbp(arr)
            else:
                raise ValueError(f"Unknown method: {method}")
            if feat is not None:
                features.append(feat)
            else:
                print(f"[FEAT SKIP] Could not extract features from {p}")
        except Exception as e:
            print(f"[FEAT SKIP] {p}: {e}")

        if (i + 1) % 100 == 0:
            print(f"Processed {i+1} / {len(paths)} images")

    features = np.stack(features, axis=0)
    labels = labels[: len(features)]  # align in case of skipped images

    np.save(f"{output_prefix}_features.npy", features)
    np.save(f"{output_prefix}_labels.npy", labels)
    print(f"Saved features to {output_prefix}_features.npy with shape {features.shape}")
    print(f"Saved labels   to {output_prefix}_labels.npy with shape {labels.shape}")


if __name__ == "__main__":
    Path("features").mkdir(exist_ok=True)

    svm_method = SVM_FEATURE_EXTRACTION_METHOD
    size = (256, 256)
    
    paths_to_features(
        csv_path="data/splits/train_paths.csv",
        output_prefix=f"features/{svm_method}/train",
        method=svm_method,
        size=size,
    )

    paths_to_features(
        csv_path="data/splits/val_paths.csv",
        output_prefix=f"features/{svm_method}/val",
        method=svm_method,
        size=size,
    )
    
    knn_method = KNN_FEATURE_EXTRACTION_METHOD
    
    paths_to_features(
        csv_path="data/splits/train_paths.csv",
        output_prefix=f"features/{knn_method}/train",
        method=knn_method,
        size=size,
    )

    paths_to_features(
        csv_path="data/splits/val_paths.csv",
        output_prefix=f"features/{knn_method}/val",
        method=knn_method,
        size=size,
    )
