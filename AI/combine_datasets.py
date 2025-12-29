import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from pathlib import Path
from sort_roboflow_dataset_train_test_valid import sort_dataset, create_data_yaml

import shutil


DATASET_1_PATH = Path("/home/samuel/Downloads/dataset_new_version/Bjornkjaervej_TestFlight_2_bigger_yolov12")
DATASET_2_PATH = Path("/home/samuel/Downloads/dataset_new_version/Bjornkjaervej_TestFlight_2_mid_yolov12")
DATASET_3_PATH = Path("/home/samuel/Downloads/dataset_new_version/Bjornkjaervej_TestFlight_2_small_yolov12")

TRAIN_PERCENT = 0.70
VAL_PERCENT = 0.25
TEST_PERCENT = 0.05

def normalize_classes_to_basic_one(label_file_path: Path):
    # Read all lines first
    with open(label_file_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()

        # skip empty lines
        if not parts:
            continue

        parts[0] = "0"  # force class to 0
        new_lines.append(" ".join(parts) + "\n")

    # Overwrite the file with updated labels
    with open(label_file_path, "w") as f:
        f.writelines(new_lines)



def rename_dataset_contents_based_on_last_folder(folder_path):
    """
    Renames images in a folder based on the root name.
    """
    folder_path = Path(folder_path)
    if not folder_path.exists():
        print(f"Folder does not exist: {folder_path}")
        return
    
    # get Bjornkjaervej_TestFlight_2_xxx from the path
    root_index = folder_path.as_posix().rfind("/") + 1
    root_name = folder_path.as_posix()[root_index:]
    print(f"Renaming images in {folder_path} based on root name: {root_name}")

    images = os.listdir(folder_path / "train" / "images")

    if len(images) == 0:
        print(f"No images found in {folder_path / 'train' / 'images'}")
        return

    for i, image_name in enumerate(images):
        image_path = folder_path / "train" / "images" / image_name
        if image_path.is_file():
            if root_name in image_name:
                print(f"Skipping {image_name}, already renamed.")
                continue
            new_image_name = f"{root_name}_{image_name}"
            new_image_path = folder_path / "train" / "images" / new_image_name
            os.rename(image_path, new_image_path)
            print(f"Renamed {image_name} to {new_image_name}")

    labels = os.listdir(folder_path / "train" / "labels")

    if len(labels) == 0:
        print(f"No labels found in {folder_path / 'train' / 'labels'}")
        return

    for i, label_name in enumerate(labels):
        label_path = folder_path / "train" / "labels" / label_name
        if label_path.is_file():
            if root_name in label_name:
                print(f"Skipping {label_name}, already renamed.")
                continue
            new_label_name = f"{root_name}_{label_name}"
            new_label_path = folder_path / "train" / "labels" / new_label_name
            os.rename(label_path, new_label_path)
            print(f"Renamed {label_name} to {new_label_name}")

def combine_datasets(dataset_paths, output_path, normalize_classes=False):

    if len(dataset_paths) < 2:
        print("At least two datasets are required to combine.")
        return
    
    for dataset_path in dataset_paths:
        if not os.path.exists(dataset_path):
            print(f"Dataset path does not exist: {dataset_path}")
            return
        
    output_path = Path(output_path)
    os.makedirs(output_path, exist_ok=True)

    combined_images_path = output_path / "images"
    combined_labels_path = output_path / "labels"
    os.makedirs(combined_images_path, exist_ok=True)
    os.makedirs(combined_labels_path, exist_ok=True)

    for dataset_path in dataset_paths:
        dataset_path = Path(dataset_path)
        images_path = dataset_path / "train" / "images"
        labels_path = dataset_path / "train" / "labels"

        if len(os.listdir(images_path)) == 0:
            print(f"No images found in {images_path}, skipping this dataset.")
            continue

        # ---- copy images ----
        for image_file in os.listdir(images_path):
            src_image_path = images_path / image_file
            dst_image_path = combined_images_path / image_file

            if os.path.isfile(src_image_path):
                shutil.copy2(src_image_path, dst_image_path)

        # ---- copy labels ----
        for label_file in os.listdir(labels_path):
            src_label_path = labels_path / label_file
            dst_label_path = combined_labels_path / label_file

            if not os.path.isfile(src_label_path):
                continue

            # First copy label
            shutil.copy2(src_label_path, dst_label_path)

            # Optionally normalize the **copied** file (so originals stay unchanged)
            if normalize_classes:
                normalize_classes_to_basic_one(dst_label_path)



if __name__ == "__main__":
    output_path = Path("/home/samuel/Downloads/dataset_new_version/combined_dataset")

    datasets = [DATASET_1_PATH, DATASET_2_PATH, DATASET_3_PATH]

    for dataset_path in datasets:
        rename_dataset_contents_based_on_last_folder(dataset_path)


    combine_datasets(datasets, output_path, normalize_classes=False)

    output_path = Path("/home/samuel/Downloads/dataset_new_version/combined_dataset_normalized")
    combine_datasets(datasets, output_path, normalize_classes=True)

    print(f"Datasets combined and saved to {output_path}")

    sort_dataset(
        dataset_path=output_path,
        output_path=output_path / "sorted",
        train_percent=TRAIN_PERCENT,
        val_percent=VAL_PERCENT,
        test_percent=TEST_PERCENT
    )
    print(f"Combined dataset sorted and saved to {output_path / 'sorted'}")

    classes = ["Potatoes"]

    create_data_yaml(
        output_path=output_path / "sorted",
        classes=classes
    )
