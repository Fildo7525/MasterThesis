from numpy import uint16
import numpy as np
import rasterio
from rasterio.enums import Resampling
from create_indexes import Bands, Indices, compute_index
import cv2
from pathlib import Path
from typing import List




def main(input_path: Path, reference_tif: Path, output_path: Path):

    img = cv2.imread(str(input_path))
    if img is None:
        print(f"[ERROR]: Could not read image from path: {input_path}")
        return

    bands = []
    for i in range(img.shape[2]):
        bands.append(img[:, :, i])

    with rasterio.open(reference_tif, "r") as src:
        profile = src.profile.copy()

    bands.reverse()
    profile.update(dtype=rasterio.uint8, count=len(bands))
    with rasterio.open(output_path, "w", **profile) as dst:
        for i in range(len(bands)):
            dst.write(bands[i], i + 1)


if __name__ == "__main__":
    input_path = Path("./tile_2_5_NRN_annotated.png")
    reference_tif = Path("./tile_2_5_NRN.tif")
    output_path = Path("./tile_2_5_NRN_annotated.tif")

    main(input_path, reference_tif, output_path)
