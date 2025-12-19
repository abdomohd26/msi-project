import csv
import random
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from PIL import Image

LABEL_MAP = {
    "glass": 0,
    "paper": 1,
    "cardboard": 2,
    "plastic": 3,
    "metal": 4,
    "trash": 5,
}

AUG_TARGETS = {
    0: 500,
    1: 500,
    2: 420,
    3: 450,
    4: 450,
    5: 300,
}


def load_image_paths(root_dir):
    paths_labels = []
    root_path = Path(root_dir)

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    for label_dir in root_path.iterdir():
        if not label_dir.is_dir():
            continue

        name = label_dir.name.lower()
        if name not in LABEL_MAP:
            print(f"[SKIP CLASS] {name}")
            continue

        label = LABEL_MAP[name]

        for file_path in label_dir.rglob('*'):
            if file_path.suffix.lower() in image_extensions:
                try:
                    Image.open(file_path).verify()
                    paths_labels.append((str(file_path), label))
                except Exception:
                    print(f"[CORRUPT] {file_path}")

    return paths_labels


def train_val_split(paths_labels, val_ratio=0.2, seed=42):
    random.seed(seed)
    by_class = defaultdict(list)

    for path, label in paths_labels:
        by_class[label].append((path, label))

    train, val = [], []

    for label, items in by_class.items():
        random.shuffle(items)
        n_val = int(len(items) * val_ratio)
        val.extend(items[:n_val])
        train.extend(items[n_val:])

    random.shuffle(train)
    random.shuffle(val)
    return train, val

def augment_img(img):
    # horizontal flip 
    if random.random() < 0.5:
        img = cv2.flip(img, 1)

    # rotation
    angle = random.uniform(-10, 10)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # brightness / contrast
    alpha = random.uniform(0.9, 1.1)
    beta = random.uniform(-15, 15)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    return img

def augment_train(train_list):
    counts = defaultdict(int)
    originals_by_label = defaultdict(list)

    for path, label in train_list:
        counts[label] += 1
        originals_by_label[label].append(path)

    augmented_samples = []

    for label, original_count in counts.items():
        target = AUG_TARGETS.get(label, original_count)
        current = original_count

        if current >= target:
            continue

        needed = target - current
        originals = originals_by_label[label]

        aug_dir = Path(f"data/augmented/{label}")
        aug_dir.mkdir(parents=True, exist_ok=True)

        for i in range(needed):
            orig_path = random.choice(originals)
            img = cv2.imread(orig_path)

            if img is None:
                continue

            aug_img = augment_img(img)
            new_name = f"{Path(orig_path).stem}_aug{i}.jpg"
            save_path = aug_dir / new_name

            cv2.imwrite(str(save_path), aug_img)
            augmented_samples.append((str(save_path), label))

    train_list.extend(augmented_samples)



if __name__ == "__main__":
    root_dir = "data/raw"

    paths_labels = load_image_paths(root_dir)

    train_paths, val_paths = train_val_split(paths_labels)

    augment_train(train_paths)

    splits_dir = Path("data/splits")
    splits_dir.mkdir(exist_ok=True)

    with open(splits_dir / "train_paths.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        writer.writerows(train_paths)

    with open(splits_dir / "val_paths.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        writer.writerows(val_paths)

    print(f"Train samples: {len(train_paths)}")
    print(f"Val samples:   {len(val_paths)}")
