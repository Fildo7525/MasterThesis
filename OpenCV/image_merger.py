#!/usr/bin/env python3

import os
import re
import rasterio
from rasterio.windows import Window
from pathlib import Path
import numpy as np
from tqdm import tqdm


def merge_tiles(tiles_dir: Path, output_tif: Path, tile_size: int, overlap: int = 0):
    """
    Merges GeoTIFF tiles back into a single image.

    Parameters
    ----------
    tiles_dir : Path
        Directory containing the tile files (tile_i_j.tif format)
    output_tif : Path
        Path to save the merged output GeoTIFF
    tile_size : int
        Size (in pixels) of each square tile used during splitting
    overlap : int, optional
        Overlap between adjacent tiles in pixels (default: 0)
    """
    tiles_dir = Path(tiles_dir)

    # Find all tile files and extract their indices
    tile_pattern = re.compile(r'tile_(\d+)_(\d+)\.tif')
    tiles = {}

    for tile_file in tiles_dir.glob('tile_*.tif'):
        match = tile_pattern.match(tile_file.name)
        if match:
            i, j = int(match.group(1)), int(match.group(2))
            tiles[(i, j)] = tile_file

    if not tiles:
        raise ValueError(f"No tiles found in {tiles_dir}")

    # Determine grid dimensions
    max_i = max(i for i, j in tiles.keys())
    max_j = max(j for i, j in tiles.keys())
    n_rows = max_i + 1
    n_cols = max_j + 1

    print(f"Found {len(tiles)} tiles in {n_rows}x{n_cols} grid")

    # Read the first tile to get metadata
    first_tile = tiles[(0, 0)]
    with rasterio.open(first_tile, "r") as src:
        tile_profile = src.profile.copy()
        n_bands = src.count
        dtype = src.dtypes[0]

        # Get the transform from the first tile
        first_transform = src.transform


    with rasterio.open(tiles[(2, 5)], "r") as src:
        description = src.descriptions

    print(f"Detected descriptions: {description}")
    # Calculate output dimensions
    output_width = n_cols * tile_size - (n_cols - 1) * overlap
    output_height = n_rows * tile_size - (n_rows - 1) * overlap

    print(f"Output image size: {output_width}x{output_height} px")
    print(f"Number of bands: {n_bands}")

    # Update profile for output
    output_profile = tile_profile.copy()
    output_profile.update({
        'height': output_height,
        'width': output_width,
        'transform': first_transform,
        "driver": "GTiff",
        "BIGTIFF": "YES",
    })

    # Create output file and write tiles
    print(f"Merging tiles into {output_tif}...")

    print(f"Output profile: {output_profile}")

    with rasterio.open(output_tif, 'w', **output_profile) as dst:
        with tqdm(total=len(tiles), desc="Merging tiles", unit="tile") as pbar:
            for (i, j), tile_path in sorted(tiles.items()):
                try:
                    # Calculate position in output image
                    x_off = j * tile_size - j * overlap
                    y_off = i * tile_size - i * overlap

                    # Read tile data
                    with rasterio.open(tile_path, "r") as src:
                        tile_data: np.ndarray = src.read()
                        h, w = tile_data.shape[1], tile_data.shape[2]

                        tile_data[tile_data > 15000] = 0  # Set no-data values to 0

                        # Write to output at correct position
                        window = Window(x_off, y_off, w, h)
                        dst.write(tile_data, window=window)

                    pbar.update(1)
                except Exception as e:
                    print(f"Error processing tile {tile_path}: {e}")

        for b in range(n_bands):
            print(f"Setting description for band {b + 1}: {description[b]}")
            dst.set_band_description(b + 1, description[b])

    print("✅ Done merging tiles!")


if __name__ == "__main__":
    # Example usage - adjust these parameters to match your splitting parameters
    tile_size = 2048
    overlap = 0

    tiles_dir = Path("../Orthomosaics/NRN_big_v2")
    output_tif = Path("./BV_TF2_NRN_big_v5.tif")

    merge_tiles(tiles_dir, output_tif, tile_size, overlap)

# 0 - 25.5314
# 0 - 2.56415
# 0 - 65534.7
