import os
import shutil
from pathlib import Path

# ---------- CONFIGURE THESE ----------
ORIGINAL_IMG_DIR = Path("/home/samuel/SDU/allimages")   # <-- YOUR GOOD PNGs
ROBOFLOW_ROOT = Path("/home/samuel/Downloads/dataset_new_version_roboflow2/combined_dataset_normalized/sorted")         # <-- Roboflow dataset root
# -------------------------------------

splits = ["train", "val", "test"]

def infer_original_name(roboflow_name: str) -> str:
    """
    Map Roboflow name back to your original PNG name.

    Example:
    Bjornkjaervej_..._yolov12_tile_2_4_NEN_png.rf.<hash>.jpg
        ->
    Bjornkjaervej_..._tile_2_4_NEN.png
    """
    base = roboflow_name.split("_png.rf")[0]
    base = base.replace("_yolov12", "")
    return base + ".txt"

def test():
    data = Path("/home/samuel/Downloads/dataset_new_version_roboflow/combined_dataset_normalized/sorted/predictions") / "run0"
    for item in data.iterdir():
        original_name = infer_original_name(item.name)
        print(f"{item.name}  -->  {original_name}")

        # rename file
        os.rename(item, data / original_name)

if __name__ == "__main__":
    test()
    # for split in splits:
    #     roboflow_image_split_dir = ROBOFLOW_ROOT / "images" / split
    #     roboflow_label_split_dir = ROBOFLOW_ROOT / "labels" / split
    #     original_split_dir = ORIGINAL_IMG_DIR

    #     # First: rename Roboflow files to match original naming scheme
    #     for roboflow_img_path in roboflow_image_split_dir.glob("*.jpg"):
    #         original_name = infer_original_name(roboflow_img_path.name)
    #         os.rename(roboflow_img_path, roboflow_image_split_dir / original_name)

    #     for roboflow_label_path in roboflow_label_split_dir.glob("*.txt"):
    #         original_name = (
    #             infer_original_name(roboflow_label_path.name.replace(".txt", ".jpg"))
    #             .replace(".jpg", ".txt")
    #         )
    #         os.rename(roboflow_label_path, roboflow_label_split_dir / original_name)

    #     # Second: replace JPGs with original PNGs
    #     for item in list(roboflow_image_split_dir.iterdir()):
    #         print(item.name)

    #         base_name = item.name.replace(".jpg", "")

    #         for original_item in original_split_dir.iterdir():
    #             if original_item.name.replace(".png", "") == base_name:

    #                 # Remove the old Roboflow JPG
    #                 if item.suffix == ".jpg":
    #                     item.unlink()   # <-- DELETE OLD FILE

    #                 # Copy the original PNG in
    #                 destination = roboflow_image_split_dir / original_item.name
    #                 shutil.copy(original_item, destination)