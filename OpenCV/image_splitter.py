#!/usr/bin/env python3

from typing import Callable
import os
import math
import rasterio
from rasterio.windows import Window
from pathlib import Path
from tqdm import tqdm
from create_indexes import Bands

from rasterio.warp import reproject,  Resampling
from affine import Affine

ORTHO_IMG_DIR = Path("../Orthomosaics/")

def process_tile(src, i, j, tile_size, output_dir, overlap, angle:float = 0.0, process_window=lambda x, i, j: x):
    # Define pixel offsets
    x_off: float = j * tile_size - j * overlap
    y_off: float = i * tile_size - i * overlap
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

    rotation = (
        Affine.translation(cx, cy) *
        Affine.rotation(angle) *
        Affine.translation(-cx, -cy)
    )

    rotated_transform = transform * rotation

    # Update profile for each tile
    tile_profile = src.profile.copy()
    tile_profile.update({
        "height": h,
        "width": w,
        "transform": rotated_transform
    })

    names = [band.name for band in Bands]

    # Output filename
    tile_name = f"tile_{i}_{j}.tif"
    out_path = os.path.join(output_dir, tile_name)

    # User defined processing function
    bands = process_window(src.read(window=window), i, j)

    # Read the window and write it out
    i = 0
    with rasterio.open(out_path, "w", **tile_profile) as dst:
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


def split_geotiff(input_tif: Path, output_dir: Path, tile_size: int, overlap: int = 0, angle: float = 0.0, process_window=lambda x, i, j: x):
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
        print(f"Splitting into {n_cols} x {n_rows} tiles")

        # Loop through tiles
        total_tiles = n_rows * n_cols
        with tqdm(total=total_tiles, desc=f"Processing {input_tif.name}", unit="tile") as pbar:
            for i in range(n_rows):
                for j in range(n_cols):
                    process_tile(src, i, j, tile_size, output_dir, overlap, angle, process_window)
                    pbar.update(1)

    print("✅ Done splitting GeoTIFF!")


if __name__ == "__main__":
    # Example usage
    tile_size = 1024
    overlap = 100
    angle = 45.0  # degrees

    print(os.listdir(ORTHO_IMG_DIR))

    output_dir = ORTHO_IMG_DIR / "example_tiles"
    split_geotiff(ORTHO_IMG_DIR / "small" / "20250827_Bjørnkjærvej_TestFlight_2_small.tif", output_dir, tile_size, overlap, 45.0)
    # for img in ORTHO_IMG_DIR.glob("*.tif"):
    #     output_dir = ORTHO_IMG_DIR / f"{img.stem}_tiles"
    #     split_geotiff(img, output_dir, tile_size, overlap)
