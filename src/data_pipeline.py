import csv
import random
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageEnhance, UnidentifiedImageError

LABEL_MAP = {
    "glass": 0,
    "paper": 1,
    "cardboard": 2,
    "plastic": 3,
    "metal": 4,
    "trash": 5,
}

def load_image_paths(root_dir):
    paths_labels = []
    root_path = Path(root_dir)

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}

    for label_dir in root_path.iterdir():
        if label_dir.is_dir():
            name = label_dir.name
            if name not in LABEL_MAP:
                print(f"[SKIP CLASS] Unknown folder name (not in LABEL_MAP): {name}")
                continue
            label = LABEL_MAP[name]

            for file_path in label_dir.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                    try:
                        Image.open(file_path).verify()
                        paths_labels.append((str(file_path), label))
                    except Exception:
                        print(f"[SKIP] Corrupted or unreadable image: {file_path}")
                        continue

    return paths_labels


def train_val_split(paths_labels, val_ratio=0.2, seed=42):
    """
    Split the paths_labels into train and validation sets.
    Returns (train_list, val_list)
    """
    random.seed(seed)
    # Shuffle the list
    shuffled = paths_labels.copy()
    random.shuffle(shuffled)
    
    # Calculate split index
    val_size = int(len(shuffled) * val_ratio)
    val_list = shuffled[:val_size]
    train_list = shuffled[val_size:]
    
    return train_list, val_list

def augment_train(train_list, target_per_class=500):
    counts = defaultdict(int)
    originals_by_label = defaultdict(list)
    for path, label in train_list:
        counts[label] += 1
        originals_by_label[label].append(path)

    for label, count in counts.items():
        if count < target_per_class:
            needed = target_per_class - count
            originals = originals_by_label[label]
            augmented_dir = Path(f"data/augmented/{label}")
            augmented_dir.mkdir(parents=True, exist_ok=True)

            for i in range(needed):
                orig_path = random.choice(originals)
                img = cv2.imread(orig_path)
                if img is None:
                    print(f"[AUG SKIP] Could not read image: {orig_path}")
                    continue

                # Apply random augmentations
                augmented_img = apply_random_augmentations(img)

                orig_stem = Path(orig_path).stem
                new_name = f"{orig_stem}_aug{i}.jpg"
                save_path = augmented_dir / new_name
                cv2.imwrite(str(save_path), augmented_img)

                train_list.append((str(save_path), label))

def apply_random_augmentations(img):
    """
    Apply a sequence of random augmentations to the image.
    """
    # Ensure at least one augmentation is applied
    augmentations_applied = 0

    # Horizontal flip
    if random.random() > 0.5:
        img = cv2.flip(img, 1)
        augmentations_applied += 1

    # Vertical flip (less common)
    if random.random() > 0.7:
        img = cv2.flip(img, 0)
        augmentations_applied += 1

    # Rotation
    if random.random() > 0.4 or augmentations_applied == 0:
        angle = random.uniform(-20, 20)
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, rot_matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
        augmentations_applied += 1

    # Brightness and contrast adjustment
    if random.random() > 0.5:
        alpha = random.uniform(0.8, 1.2)  # contrast
        beta = random.uniform(-30, 30)    # brightness
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        augmentations_applied += 1

    # Gaussian blur
    if random.random() > 0.6 and augmentations_applied < 3:
        kernel_size = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        augmentations_applied += 1

    # Add Gaussian noise
    # if random.random() > 0.7 and augmentations_applied < 3:
    #     noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
    #     img = cv2.add(img, noise)
    #     augmentations_applied += 1

    # Random crop and resize (zoom effect)
    # if random.random() > 0.5 and augmentations_applied < 3:
    #     h, w = img.shape[:2]
    #     crop_factor = random.uniform(0.8, 1.0)
    #     new_h, new_w = int(h * crop_factor), int(w * crop_factor)
    #     start_h = random.randint(0, h - new_h)
    #     start_w = random.randint(0, w - new_w)
    #     img = img[start_h:start_h+new_h, start_w:start_w+new_w]
    #     img = cv2.resize(img, (w, h))
    #     augmentations_applied += 1

    # Color jitter
    if random.random() > 0.5 and augmentations_applied < 3:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= random.uniform(0.8, 1.2)  # saturation
        hsv[:, :, 0] += random.uniform(-10, 10)  # hue
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        augmentations_applied += 1

    return img


if __name__ == "__main__":
    # Load paths
    root_dir = "data/raw"
    paths_labels = load_image_paths(root_dir)

    # Split
    train_paths, val_paths = train_val_split(paths_labels)

    # Augment train data to balance classes
    augment_train(train_paths, target_per_class=500)

    # Save to CSV files
    splits_dir = Path("data/splits")
    splits_dir.mkdir(exist_ok=True)

    # Train CSV
    with open(splits_dir / "train_paths.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])  # header
        for path, label in train_paths:
            writer.writerow([path, label])

    # Val CSV
    with open(splits_dir / "val_paths.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])  # header
        for path, label in val_paths:
            writer.writerow([path, label])

    print(f"Saved {len(train_paths)} train samples and {len(val_paths)} val samples.")