#!/usr/bin/env python3

import os
import sys
import math
import rasterio
from rasterio.windows import Window
from pathlib import Path
import numpy as np
from argparse import ArgumentParser

ORTHO_IMG_DIR = Path("/home/samuel/Documents/code/Orthomosaics/")

def proceess_tile(src, i, j, tile_size, output_dir):
    # Define pixel offsets
    x_off = j * tile_size
    y_off = i * tile_size
    width = src.width
    height = src.height

    # Define window size (handle edge cases at borders)
    w = min(tile_size, width - x_off)
    h = min(tile_size, height - y_off)

    window = Window(x_off, y_off, w, h)

    # Adjust the transform for this tile
    transform = src.window_transform(window)

    # Update profile for each tile
    tile_profile = src.profile.copy()
    tile_profile.update({
        "height": h,
        "width": w,
        "transform": transform
    })

    # Output filename
    tile_name = f"tile_{i}_{j}.tif"
    out_path = os.path.join(output_dir, tile_name)

    # Read the window and write it out
    with rasterio.open(out_path, "w", **tile_profile) as dst:
        dst.write(src.read(window=window))

    print(f"Saved: {out_path}")


def split_geotiff(input_tif: Path, output_dir: Path, tile_size: int):
    """
    Splits a multispectral GeoTIFF into square tiles.

    Parameters
    ----------
    input_tif : str
        Path to the input GeoTIFF file (.tif)
    output_dir : str
        Directory to save the output tiles
    tile_size : int
        Size (in pixels) of each square tile
    """

    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    print(f"Splitting {input_tif} into tiles of size {tile_size}x{tile_size} px...")

    # Open the source GeoTIFF
    with rasterio.open(input_tif) as src:
        width = src.width
        height = src.height

        print(f"Source profile: {src.profile}")

        # Number of tiles horizontally and vertically
        n_cols = math.ceil(width / tile_size)
        n_rows = math.ceil(height / tile_size)

        print(f"Image size: {width}x{height} px")
        print(f"Tile size: {tile_size} px")
        print(f"Splitting into {n_cols} x {n_rows} tiles")

       # Loop through tiles
        for i in range(n_rows):
            for j in range(n_cols):
                proceess_tile(src, i, j, tile_size, output_dir)

    print("✅ Done splitting GeoTIFF!")

if __name__ == "__main__":
    # Example usage
    tile_size = 1024

    print(os.listdir(ORTHO_IMG_DIR))

    for img in ORTHO_IMG_DIR.glob("*.tif"):
        output_dir = ORTHO_IMG_DIR / f"{img.stem}_tiles"
        split_geotiff(img, output_dir, tile_size)

