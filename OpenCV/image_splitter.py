#!/usr/bin/env python3

from typing import Callable, Any
import os
import math
import rasterio
from rasterio.io import DatasetReader
from rasterio.windows import Window
from pathlib import Path
from tqdm import tqdm
from create_indexes import Bands
import numpy as np

from rasterio.warp import reproject,  Resampling
from affine import Affine

from rasterio.warp import reproject,  Resampling
from affine import Affine

ORTHO_IMG_DIR = Path("../Orthomosaics/")

def process_tile(
        src, i: int, j: int,
        tile_size: int,
        output_dir: Path,
        overlap: int = 0,
        angle:float = 0.0,
        offset: tuple = (0, 0),
        save: bool = False,
        process_window: Callable[[DatasetReader, Window, int, int], None | np.ndarray] = lambda src, window, i, j: np.array([])):

    # Define pixel offsets
    x_off: float = j * tile_size - j * overlap + offset[0]
    y_off: float = i * tile_size - i * overlap + offset[1]
    width: float = src.width
    height: float = src.height

    # Define window size (handle edge cases at borders)
    w = min(tile_size, width - x_off)
    h = min(tile_size, height - y_off)

    if x_off < 0 or y_off < 0 or w <= 0 or h <= 0:
        # print(f"Skipping processing tile ({i}, {j}) at offset ({x_off}, {y_off}) with size ({w}, {h})")
        return  # Skip tiles that are out of bounds

    # print(f"Processing tile ({i}, {j}) at offset ({x_off}, {y_off}) with size ({w}, {h})")
    window = Window(x_off, y_off, w, h)

    # Adjust the transform for this tile
    transform = src.window_transform(window)

    # Output filename
    tile_name = f"tile_{i}_{j}.tif"
    out_path = os.path.join(output_dir, tile_name)

    # Tile center in pixel coordinates (of the tile itself)
    cx = w / 2
    cy = h / 2

    rotation: Affine = (
        Affine.translation(cx, cy) *
        Affine.rotation(angle) *
        Affine.translation(-cx, -cy)
    )

    rotated_transform = transform * rotation
    new_transform = rotated_transform if angle != 0.0 else transform

    # Update profile for each tile
    tile_profile = src.profile.copy()
    tile_profile.update({
        "height": h,
        "width": w,
        "transform": new_transform,
    })

    # print(f"Source description: {src.descriptions or "No descriptions"}")
    names = src.descriptions if src.descriptions is not None else [band.name for band in Bands]

    # Output filename
    tile_name = f"tile_{i}_{j}.tif"
    out_path = os.path.join(output_dir, tile_name)

    # User defined processing function
    process_window(src, window, i, j)
    if not save:
        return

    bands = src.read(window=window)

    # Read the window and write it out
    with rasterio.open(out_path, "w", **tile_profile) as dst:
        if angle == 0.0:
            dst.write(bands)
            for idx, name in enumerate(names, start=1):
                # print(f"Setting band {idx} description to {name}")
                dst.set_band_description(idx, name)

            # return

        else:
            for band in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band),
                    destination=rasterio.band(dst, band),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=rotated_transform,
                    dst_crs=src.crs,
                    src_window=window,
                    resampling=Resampling.bilinear
                )

            x_min, y_min, x_max, y_max = src.window_bounds(window)
            dst.update_tags(
                ORIGINAL_X_MIN=x_min,
                ORIGINAL_Y_MIN=y_min,
                ORIGINAL_X_MAX=x_max,
                ORIGINAL_Y_MAX=y_max,
                ROTATION_ANGLE=angle,
            )


def split_geotiff(input_tif: Path,
        output_dir: Path,
        tile_size: int,
        overlap: int = 0,
        angle: float = 0.0,
        offset: tuple = (0, 0),
        save: bool = True,
        process_window: Callable[[DatasetReader, Window, int, int], np.ndarray | None] = lambda src, window, i, j: np.array([])):
    """
    Splits a multispectral GeoTIFF into square tiles with optional overlap.

    Parameters
    ----------
    input_tif : Path
        Path to the input GeoTIFF file (.tif)
    output_dir : Path
        Directory to save the output tiles
    tile_size : int
        Size (in pixels) of each square tile
    overlap : int, optional
        Overlap between adjacent tiles in pixels (default: 0)
    angle : float [degrees] (default: 0.0)
        Rotation angle in **degrees** to apply to each tile (default: 0.0)
    offset : tuple, optional
        Offset in pixels to apply to the final merged image (default: (0, 0))
    process_window : Callable, optional
        A function to process each window of the image. It should accept the window data and the tile indices (i, j) and return the processed window data. Default is a no-op function.
    """
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    print(f"Splitting {input_tif} into tiles of size {tile_size}x{tile_size} px with {overlap} px overlap...")

    # Open the source GeoTIFF
    with rasterio.open(input_tif) as src:
        width = src.width
        height = src.height

        print(f"Source profile: {src.profile}")

        # Number of tiles horizontally and vertically
        n_cols = math.ceil((width - overlap) / (tile_size - overlap))
        n_rows = math.ceil((height - overlap) / (tile_size - overlap))

        print(f"Image size: {width}x{height} px")
        print(f"Tile size: {tile_size} px")
        print(f"Overlap: {overlap} px")
        print(f"angle: {angle} degrees")
        print(f"Offset x: {offset[0]} px")
        print(f"Offset y: {offset[1]} px")
        print(f"Splitting into {n_cols} x {n_rows} tiles")

        # Loop through tiles
        total_tiles = n_rows * n_cols
        with tqdm(total=total_tiles, desc=f"Processing {input_tif.name}", unit="tile") as pbar:
            for i in range(n_rows):
                for j in range(n_cols):
                    try:
                        process_tile(src, i, j, tile_size, output_dir, overlap, angle, offset, save, process_window)
                        pbar.update(1)
                    except Exception as e:
                        print(f"Error processing tile ({i}, {j}): {e}")

    print("✅ Done splitting GeoTIFF!")


if __name__ == "__main__":
    # Example usage
    tile_size = 1024
    overlap = 0
    angle = 0  # degrees
    offset = (0, 0)  # pixels

    orthomosaic = ORTHO_IMG_DIR / "20250827_Bjørnkjærvej_TestFlight_2_bigger_v2.tif"
    output_dir = ORTHO_IMG_DIR / "example_tiles_big"

    split_geotiff(
        input_tif = orthomosaic,
        output_dir = output_dir,
        tile_size = tile_size,
        overlap = overlap,
        angle = angle,
        offset = offset,
    )
    # for img in ORTHO_IMG_DIR.glob("*.tif"):
    #     output_dir = ORTHO_IMG_DIR / f"{img.stem}_tiles"
    #     split_geotiff(img, output_dir, tile_size, overlap)
