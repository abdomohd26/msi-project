import csv
import random
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
                try:
                    image = Image.open(orig_path)
                except UnidentifiedImageError:
                    print(f"[AUG SKIP] Bad image during augment: {orig_path}")
                    continue

                transforms_applied = 0

                # horizontal flip
                if random.random() > 0.5:
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                    transforms_applied += 1

                # # rotation
                # if random.random() > 0.5 or transforms_applied == 0:
                #     angle = random.uniform(-15, 15)
                #     image = image.rotate(angle)
                #     transforms_applied += 1

                # brightness
                if random.random() > 0.5 and transforms_applied < 2:
                    enhancer = ImageEnhance.Brightness(image)
                    image = enhancer.enhance(random.uniform(0.8, 1.2))
                    transforms_applied += 1

                # contrast
                if random.random() > 0.5 and transforms_applied < 2:
                    enhancer = ImageEnhance.Contrast(image)
                    image = enhancer.enhance(random.uniform(0.8, 1.2))
                    transforms_applied += 1

                orig_stem = Path(orig_path).stem
                new_name = f"{orig_stem}_aug{i}.jpg"
                save_path = augmented_dir / new_name
                image.save(save_path)

                train_list.append((str(save_path), label))


if __name__ == "__main__":
    # Load paths
    root_dir = "data/raw"
    paths_labels = load_image_paths(root_dir)

    # Split
    train_paths, val_paths = train_val_split(paths_labels)

    # Augment train data to balance classes
    augment_train(train_paths)

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