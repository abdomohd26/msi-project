import csv
from pathlib import Path

import numpy as np
from PIL import Image
from skimage import color, filters, measure
from skimage.feature import hog, local_binary_pattern


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


def image_to_feature_flat(path, size=(64, 64)):
    """
    Resize -> RGB -> flatten to 1D vector.
    """
    arr = image_to_array(path, size=size)
    return arr.flatten()  # shape: (H*W*3,)


def image_to_feature_hist_grad(path, size=(64, 64), bins=16):
    """
    Resize -> RGB & HSV -> per-channel histograms + gradient mean/std.
    Returns 1D feature vector.
    """
    arr = image_to_array(path, size=size)  # [H, W, 3] in [0,1]

    # RGB histograms
    r = arr[:, :, 0].ravel()
    g = arr[:, :, 1].ravel()
    b = arr[:, :, 2].ravel()

    r_hist, _ = np.histogram(r, bins=bins, range=(0.0, 1.0), density=True)
    g_hist, _ = np.histogram(g, bins=bins, range=(0.0, 1.0), density=True)
    b_hist, _ = np.histogram(b, bins=bins, range=(0.0, 1.0), density=True)

    # HSV histograms
    hsv = color.rgb2hsv(arr)
    h = hsv[:, :, 0].ravel()
    s = hsv[:, :, 1].ravel()
    v = hsv[:, :, 2].ravel()

    h_hist, _ = np.histogram(h, bins=bins, range=(0.0, 1.0), density=True)
    s_hist, _ = np.histogram(s, bins=bins, range=(0.0, 1.0), density=True)
    v_hist, _ = np.histogram(v, bins=bins, range=(0.0, 1.0), density=True)

    # Gradient magnitude (on grayscale)
    gray = color.rgb2gray(arr)
    grad = filters.sobel(gray)
    grad_mean = np.mean(grad)
    grad_std = np.std(grad)

    # Concatenate: 3*RGB + 3*HSV histograms + 2 gradient stats
    feature = np.concatenate(
        [r_hist, g_hist, b_hist, h_hist, s_hist, v_hist, [grad_mean, grad_std]]
    )
    return feature  # length = 6*bins + 2 (e.g., 98 if bins=16)

def image_to_feature_advanced(path, size=(64, 64), bins=16):
    """
    Safe features: RGB/HSV histograms + gradient stats + LBP only.
    Total 157 dims - optimal for SVM on 3000 samples.
    """
    arr = image_to_array(path, size=size)  # [H, W, 3] in [0,1]

    # RGB histograms (48 bins)
    r = arr[:, :, 0].ravel()
    g = arr[:, :, 1].ravel()
    b = arr[:, :, 2].ravel()
    r_hist, _ = np.histogram(r, bins=bins, range=(0.0, 1.0), density=True)
    g_hist, _ = np.histogram(g, bins=bins, range=(0.0, 1.0), density=True)
    b_hist, _ = np.histogram(b, bins=bins, range=(0.0, 1.0), density=True)

    # HSV histograms (48 bins)
    hsv = color.rgb2hsv(arr)
    h = hsv[:, :, 0].ravel()
    s = hsv[:, :, 1].ravel()
    v = hsv[:, :, 2].ravel()
    h_hist, _ = np.histogram(h, bins=bins, range=(0.0, 1.0), density=True)
    s_hist, _ = np.histogram(s, bins=bins, range=(0.0, 1.0), density=True)
    v_hist, _ = np.histogram(v, bins=bins, range=(0.0, 1.0), density=True)

    # Gradient stats (2 dims)
    gray = color.rgb2gray(arr)
    grad = filters.sobel(gray)
    grad_mean = np.mean(grad)
    grad_std = np.std(grad)

    # LBP texture only (59 bins uniform patterns)
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 60), density=True)

    # Concat: 48(RGB)+48(HSV)+2(grad)+59(LBP) = 157 dims ✓
    feature = np.concatenate([
        r_hist, g_hist, b_hist, 
        h_hist, s_hist, v_hist, 
        [grad_mean, grad_std],
        lbp_hist
    ])
    
    return feature

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
            if method == "flat":
                feat = image_to_feature_flat(p, size=size)
            elif method == "hist_grad":
                feat = image_to_feature_hist_grad(p, size=size, bins=bins)
            elif method == "advanced":
                feat = image_to_feature_advanced(p, size=size)
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
    # Example usage:
    Path("features").mkdir(exist_ok=True)

    # Train features with advanced method
    paths_to_features(
        csv_path="data/splits/train_paths.csv",
        output_prefix="features/train",
        method="advanced",   # hist_grad or advanced
        size=(128, 128),
    )

    # Val features
    paths_to_features(
        csv_path="data/splits/val_paths.csv",
        output_prefix="features/val",
        method="advanced",   # hist_grad or advanced
        size=(128, 128),
    )
