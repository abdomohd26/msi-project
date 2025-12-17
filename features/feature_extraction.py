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
    image_to_feature_cnn,
    image_to_feature_cnn_lbp,
)


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

# ---------- Batch conversion and saving ----------

def paths_to_features(csv_path, output_prefix, method="hist_grad", size=(64, 64), bins=16):
    """
    Read paths+labels from CSV, compute features, save:
      output_prefix_features.npy
      output_prefix_labels.npy
    method: "flat", "hist_grad", or "advanced"
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
            elif method == "cnn":
                feat = image_to_feature_cnn(arr)
            elif method == "cnn_lbp":
                feat = image_to_feature_cnn_lbp(arr)
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

    method = "cnn_lbp"  # flat, hist_grad, advanced, cnn, cnn_lbp
    size = (256, 256)
    
    # Train features with advanced method
    paths_to_features(
        csv_path="data/splits/train_paths.csv",
        output_prefix="features/train",
        method=method,
        size=size,
    )

    # Val features
    paths_to_features(
        csv_path="data/splits/val_paths.csv",
        output_prefix="features/val",
        method=method,
        size=size,
    )
