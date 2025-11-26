#!/usr/bin/env python3
import os
import shutil
import random
from pathlib import Path

# --- CONFIG ---
in_dir = Path("./NEN")
image_dir = in_dir / "images" / "train"
label_dir = in_dir / "labels" / "train"

# Output folders
out = Path("./NEN_sorted")
out_images = out / "images"
out_labels = out / "labels"
Path.mkdir(out, exist_ok=True)

train_ratio = 0.85
val_ratio = 0.1
test_ratio = 0.05

# Ensure output dirs exist
for split in ["train", "val", "test"]:
    (out_images / split).mkdir(parents=True, exist_ok=True)
    (out_labels / split).mkdir(parents=True, exist_ok=True)

# Gather all images
image_files = sorted([f for f in image_dir.glob("*.*") if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".tif"]])
print(f"Found {len(image_files)} images.")

# Shuffle
random.shuffle(image_files)

# Split sizes
n = len(image_files)
train_end = int(n * train_ratio)
val_end = train_end + int(n * val_ratio)

splits = {
    "train": image_files[:train_end],
    "val": image_files[train_end:val_end],
    "test": image_files[val_end:]
}

# Copy files
for split_name, files in splits.items():
    for img in files:
        # Copy image
        shutil.copy(img, out_images / split_name / img.name)

        # Find label
        label_path = label_dir / (img.stem + ".txt")
        if label_path.exists():
            shutil.copy(label_path, out_labels / split_name / label_path.name)
        else:
            print(f"Warning: label missing for {img.name}")

dataset_yaml = out / "dataset.yaml"
with open(dataset_yaml, "w") as f:
    f.write(f"""
path: {out}
train: images/train
val: images/val
test: images/test
nc: 1
names:
  0: potato
    """)

print("Dataset split complete!")
print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
