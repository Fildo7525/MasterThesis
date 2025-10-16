import rasterio
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2
from rasterio.windows import Window
import os
from scipy.spatial import distance
from enum import IntEnum
from create_indexes import Indices, compute_index, Bands

PATH_TO_INPUT_FOLDER =  Path("/home/samuel/Documents/code").expanduser()

def prepare_images(input_path, output_path, chunk_size=512):

    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        num_bands = src.count

        print(f"Input file has {num_bands} bands")
        print(f"Datatype: {src.dtypes[0]}")
        print(f"Dimensions: {src.width} x {src.height}")

        profile.update(
        count=1,
        dtype='float32',
        compress='deflate'   # optional: smaller file
        )

        for index in Indices:
            print(f"Calculating {index.name}...")
            out_path = output_path / f"{input_path.stem}_{index.name.lower()}.tif"

            with rasterio.open(out_path, 'w', **profile) as dst:

                corrupted_chunks = 0

                for row_start in range(0, src.height, chunk_size):
                    # Define window (height-limited slice)
                    num_rows = min(chunk_size, src.height - row_start)
                    window = Window(0, row_start, src.width, num_rows)

                    try:
                        bands = [src.read(band.value, window=window).astype(np.float32) for band in Bands]

                        img_index = compute_index(index.name, bands)
                        dst.write(img_index, 1, window=window)

                    except Exception as e:
                        corrupted_chunks += 1
                        print(f"⚠️ Failed to process rows {row_start}:{row_start+num_rows} — {e}")
                        zeros = np.zeros((num_rows, src.width), dtype=np.float32)
                        dst.write(zeros, 1, window=window)

                print(f"\nProcessing complete!")
                if corrupted_chunks > 0:
                    print(f"Warning: {corrupted_chunks} corrupted chunks were replaced with zeros")

if __name__ == "__main__":
    
    output_dir = PATH_TO_INPUT_FOLDER
    image_tif = PATH_TO_INPUT_FOLDER / "20250827_Bjørnkjærvej_TestFlight_2_small.tif"
    img = prepare_images(image_tif, output_dir, chunk_size = 1024)