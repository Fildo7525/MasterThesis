import os
import cv2 as cv
import numpy as np
import rasterio
from pathlib import Path
from tqdm import tqdm
from image_processor import ImageProcessor, Bands, ThreeBandInputPaths



INPUT_DIR_LARGE = Path("/home/samuel/SDU/MasterThesis/Orthomosaics/large/processed_output")
INPUT_DIR_MID = Path("/home/samuel/SDU/MasterThesis/Orthomosaics/mid/processed_output")
INPUT_DIR_SMALL = Path("/home/samuel/SDU/MasterThesis/Orthomosaics/small/processed_output")

OUTPUT_DIR = Path("/home/samuel/SDU/MasterThesis/Orthomosaics/band_combinations") / "float_combinations"
if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR_LARGE = OUTPUT_DIR / "large"
OUTPUT_DIR_MID = OUTPUT_DIR / "mid"
OUTPUT_DIR_SMALL = OUTPUT_DIR / "small"

# PATHS = [[INPUT_DIR_LARGE, OUTPUT_DIR_LARGE], [INPUT_DIR_MID, OUTPUT_DIR_MID], [INPUT_DIR_SMALL, OUTPUT_DIR_SMALL]]

PATHS = [[INPUT_DIR_LARGE, OUTPUT_DIR_LARGE],[INPUT_DIR_MID, OUTPUT_DIR_MID], [INPUT_DIR_SMALL, OUTPUT_DIR_SMALL]]
if not OUTPUT_DIR_LARGE.exists():
    OUTPUT_DIR_LARGE.mkdir(parents=True, exist_ok=True)
if not OUTPUT_DIR_MID.exists():
    OUTPUT_DIR_MID.mkdir(parents=True, exist_ok=True)
if not OUTPUT_DIR_SMALL.exists():
    OUTPUT_DIR_SMALL.mkdir(parents=True, exist_ok=True)

proc = ImageProcessor(INPUT_DIR_MID, OUTPUT_DIR_LARGE, "")

# RED_NIR_NDVI combination (nir, red, ndvi)
ending = "RED_NIR_NDVI"

for index, path in enumerate(PATHS):

    input_dir = path[0]
    OUTPUT_DIR = path[1]
    dir = Path(str(OUTPUT_DIR) + "/" + ending)

    if not dir.exists():
        proc.set_input_path(input_dir / "image_tiles")

        for img_name in tqdm(sorted(os.listdir(proc.input_path)), desc="RED_NIR_NDVI image creation"):
            if "xml" in img_name:
                continue
            img_name_nir = img_name.replace(".png", "_NIR.png")
            img_name_er = img_name.replace(".png", "_EXTEND_RED.png")
            img_name_ndvi = img_name.replace(".png", "_ndvi.png")
            proc.calculate_three_band_image(
                input_paths = ThreeBandInputPaths(
                    band1=input_dir / "extended_red" / img_name_er,
                    band2=input_dir / "nir" / img_name_nir,
                    band3=input_dir / "image_tiles_indeces_utf8" / "NDVI" / img_name_ndvi,
                ),
                output_path=dir,
                ending=ending
            )

    else:
        print("RED_NIR_NDVI image creation skipped; output directory already exists.")