import numpy as np
from pathlib import Path
import os

def create_data_yaml(output_path: Path, classes):
    """
    Creates a dataset.yaml file for the sorted dataset.
    """

    dataset_yaml = output_path / "dataset.yaml"
    with open(dataset_yaml, "w") as f:
        f.write(
f"""
path: {output_path}
train: images/train
val: images/val
test: images/test
nc: 1
names: {classes}
"""
)
    print(f"Created dataset.yaml at {dataset_yaml}")


import os
import numpy as np
from pathlib import Path
import shutil
import json

def apply_split(
    dataset_path: Path,
    output_path: Path,
    split_file: Path,
    extension: str = ".png"
):
    with open(split_file) as f:
        split = json.load(f)

    for split_name, stems in split.items():
        img_out = output_path  / "images" / split_name
        lbl_out = output_path / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for stem in stems:
            img = dataset_path / "images" / f"{stem}{extension}"
            lbl = dataset_path / "labels" / f"{stem}.txt"

            if img.exists() and lbl.exists():
                shutil.copy(img, img_out / img.name)
                shutil.copy(lbl, lbl_out / lbl.name)


def create_split_index(
    images_dir: Path,
    labels_dir: Path,
    train_pct=0.7,
    val_pct=0.2,
    test_pct=0.1,
    seed=42,
    require_non_empty_val=True,
    extension: str = ".png"
):
    images = sorted(images_dir.glob(f"*{extension}"))  # adjust extension if needed

    # Pair image ↔ label by stem
    pairs = []
    for img in images:
        lbl = labels_dir / f"{img.stem}.txt"
        if lbl.exists():
            pairs.append(img.stem)

    rng = np.random.default_rng(seed)
    rng.shuffle(pairs)

    n = len(pairs)
    print(f"Total samples found: {n}")
    n_train = int(n * train_pct)
    n_val = int(n * val_pct)

    if require_non_empty_val:
        non_empty = [
            s for s in pairs
            if (labels_dir / f"{s}.txt").stat().st_size > 0
        ]
    else:
        non_empty = pairs

    if require_non_empty_val and len(non_empty) > n_val:
        val_stems = non_empty[:n_val]
        train_stems = [s for s in pairs if s not in val_stems][:n_train]
        test_stems = [s for s in pairs if s not in val_stems and s not in train_stems]
    else:
        val_stems = pairs[n_train:n_train + n_val]
        train_stems = pairs[:n_train]
        test_stems = pairs[n_train + n_val:]

    split = {
        "train": train_stems,
        "val": val_stems,
        "test": test_stems
    }

    return split

def sort_dataset(dataset_path: Path, output_path: Path, train_percent: float, val_percent: float, test_percent: float):
    """
    Sorts a dataset into train, validation, and test sets based on given percentages.
    Ensures validation set contains only images with non-empty labels.
    """

    dataset_images = sorted(os.listdir(dataset_path / "images"))
    dataset_labels = sorted(os.listdir(dataset_path / "labels"))

    if len(dataset_images) == 0:
        print(f"No images found in {dataset_path / 'images'}")
        return
    if len(dataset_labels) == 0:
        print(f"No labels found in {dataset_path / 'labels'}")
        return
    if len(dataset_images) != len(dataset_labels):
        print("Number of images and labels do not match!")
        return

    # Shuffle dataset
    combined = list(zip(dataset_images, dataset_labels))
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(combined)

    # Separate non-empty labels
    non_empty = [(img, lbl) for img, lbl in combined if os.path.getsize(dataset_path / "labels" / lbl) > 0]
    empty = [(img, lbl) for img, lbl in combined if os.path.getsize(dataset_path / "labels" / lbl) == 0]

    total_images = len(combined)
    train_end = int(total_images * train_percent)
    val_end = train_end + int(total_images * val_percent)

    # Assign splits
    train = combined[:train_end]
    val = non_empty[:int(total_images * val_percent)]  # only non-empty labels in val
    test = combined[val_end:]

    splits = {
        "train": train,
        "val": val,
        "test": test
    }

    # Create directories and move files
    for split_name, items in splits.items():
        split_image_path = output_path / "images" / split_name
        split_label_path = output_path / "labels" / split_name

        os.makedirs(split_image_path, exist_ok=True)
        os.makedirs(split_label_path, exist_ok=True)

        # Clear existing contents
        for folder in [split_image_path, split_label_path]:
            for f in os.listdir(folder):
                (folder / f).unlink()

        # Copy files
        for img, lbl in items:
            shutil.copy(dataset_path / "images" / img, split_image_path / img)
            shutil.copy(dataset_path / "labels" / lbl, split_label_path / lbl)

    print(f"Dataset sorted into train, val, and test sets at {output_path}")

    
