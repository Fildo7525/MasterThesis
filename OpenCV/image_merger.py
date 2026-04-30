#!/usr/bin/env python3

import os
import re
import rasterio
from rasterio.windows import Window
from pathlib import Path
import numpy as np
from tqdm import tqdm

def merge_tiles_dirs(tiles_dir: list[Path], output_tif: Path, tile_size: int, overlap: int = 0, glob="tile_*.tif", regex=r'tile_(\d+)_(\d+)\.tif'):
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
    # Find all tile files and extract their indices
    tile_pattern = re.compile(regex)
    # tile_pattern = re.compile(r'tile_(\d+)_(\d+)\.tif')
    tiles = {}

    for dir in tiles_dir:
        print(f"Scanning directory: {dir}")
        for tile_file in dir.glob(glob):
            match = tile_pattern.match(tile_file.name)
            if match:
                i, j = int(match.group(1)), int(match.group(2))
                tiles[(i, j)] = tile_file

        if not tiles:
            raise ValueError(f"No tiles found in {dir} with glob {glob} and regex {regex}")

    # Determine grid dimensions
    max_i = max(i for i, j in tiles.keys())
    max_j = max(j for i, j in tiles.keys())
    n_rows = max_i + 1
    n_cols = max_j + 1

    print(f"Found {len(tiles)} tiles in {n_rows}x{n_cols} grid")

    # Read the first tile to get metadata
    min_dist = 10**6;
    min_idx = None

    if tiles.get((0,0)) == None:
        for i, j in tiles.keys():
            d = (i**i + j**j)**0.5
            if min_dist > d:
                min_dist = d
                min_idx = (i, j)
    else:
        min_dist = (0,0)

    print(f"Setting the profile of tile {min_idx}")
    first_tile = tiles[min_idx]
    with rasterio.open(first_tile, "r") as src:
        tile_profile = src.profile.copy()
        n_bands = src.count
        dtype = src.dtypes[0]

        # Get the transform from the first tile
        first_transform = src.transform


    with rasterio.open(tiles[list(tiles.keys())[0]], "r") as src:
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
    # Find all tile files and extract their indices
    tile_pattern = re.compile(r'tile_(\d+)_(\d+)\.tif')
    tiles = {}

    print(f"Searching directory: {tiles_dir}")
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

                        # tile_data[tile_data > 15000] = 0  # Set no-data values to 0

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
    tile_size = 1024
    overlap = 0

    # tiles_dir = Path("../Orthomosaics/NRN_big_v2")
    # output_tif = Path("./BV_TF2_NRN_big_v5.tif")

    # merge_tiles(tiles_dir, output_tif, tile_size, overlap)

    base_dir = Path.home() / "SDU/MasterThesis/OpenCV/dataset_NIR_RED_NGRDI_NEW_filip_obb/images"

    tiles_dirs = [base_dir / "test", base_dir / "train", base_dir / "val"]
    globs = ['small_tile_*.tif', 'mid_tile_*.tif', 'large_tile_*.tif']
    output_tifs = [base_dir / "small.tif", base_dir / "mid.tif", base_dir / "large.tif", ]
    regexes = [r'small_tile_(\d+)_(\d+)__original.tif', r'mid_tile_(\d+)_(\d+)__original.tif', r'large_tile_(\d+)_(\d+)__original.tif']
    for glob, out_tif, regex in zip(globs, output_tifs, regexes):
        merge_tiles_dirs(tiles_dirs, out_tif, tile_size, glob=glob, regex=regex)



# 0 - 25.5314
# 0 - 2.56415
# 0 - 65534.7
