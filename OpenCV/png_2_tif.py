from numpy import uint16
import numpy as np
import rasterio
from rasterio.enums import Resampling
from create_indexes import Bands, Indices, compute_index
import cv2
from pathlib import Path
from typing import List


def png2tif(input_path: Path, reference_tif: Path, output_path: Path):
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


def tif2png(input_path: Path, output_path: Path):
    with rasterio.open(input_path, "r") as src:
        all_bands = src.read()
        out_mat = []

        for i in range(all_bands.shape[0]):
            band = all_bands[i, :, :].astype(np.float32)

            # Min–max normalization
            min_v = band.min()
            max_v = band.max()

            if max_v > min_v:
                band = (band - min_v) / (max_v - min_v)
            else:
                band = np.zeros_like(band)

            band = (band * 255).astype(np.uint8)
            out_mat.append(band)

        out_mat.reverse()
        img = np.stack(out_mat, axis=0)
        img = img.transpose(1, 2, 0)

        print(f"Read TIF with shape: {img.shape}, dtype: {img.dtype}")
        cv2.imwrite(str(output_path), img)



if __name__ == "__main__":
    # input_path = Path("./tile_2_5_NRN_annotated.png")
    # reference_tif = Path("./tile_2_5_NRN.tif")
    # output_path = Path("./tile_2_5_NRN_annotated.tif")

    # png2tif(input_path, reference_tif, output_path)

    input_path = Path("/home/fildo/SDU/MasterThesis/OpenCV/annotated_pngs/mid/tile_15_9.tif")
    output_path = Path("/home/fildo/SDU/MasterThesis/OpenCV/annotated_pngs/mid/tile_15_9.png")
    tif2png(input_path, output_path)
