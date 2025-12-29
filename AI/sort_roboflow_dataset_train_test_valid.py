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

    
