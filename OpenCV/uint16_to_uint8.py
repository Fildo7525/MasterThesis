#!/usr/bin/env python3

from pathlib import Path
ORTHO_IMG_DIR = Path("../Orthomosaics/")

import numpy as np
import rasterio
from rasterio.windows import Window
from enum import IntEnum

class Bands(IntEnum):
    RED = 1
    GREEN = 2
    BLUE = 3
    EXTEND_GREEN = 4
    EXTEND_RED = 5
    REDEDGE = 6
    NIR = 7

def convert_uint16_to_uint8(data):
    """
    Convert uint16 to uint8 with proper scaling to avoid artifacts.
    Uses percentile-based normalization for better contrast.
    """
    # Calculate 2nd and 98th percentiles to ignore outliers
    p2, p98 = np.percentile(data, (2, 98))

    # Clip values to percentile range
    data_clipped = np.clip(data, p2, p98)

    # Scale to 0-255 range
    if p98 > p2:
        data_scaled = ((data_clipped - p2) / (p98 - p2) * 255).astype(np.uint8)
    else:
        data_scaled = np.zeros_like(data, dtype=np.uint8)

    return data_scaled


def get_max_min_from_sampes(src, chunk_size, rgb_bands, percentiles=(0, 100)):
    # Read RGB data in chunks to calculate global percentiles
    print("Calculating global percentiles for scaling...")
    samples = []
    step = max(src.height // 10, 1)  # Sample ~10 rows
    corrupted_count = 0

    for i in range(0, src.height, step):
        window = Window(0, i, src.width, min(chunk_size, src.height - i))
        for band_idx in rgb_bands:
            try:
                band_data = src.read(band_idx, window=window)
                samples.append(band_data.flatten()[::100])  # Subsample for speed
            except Exception as e:
                corrupted_count += 1
                print(f"\nWarning: Skipping corrupted tile at row {i} (band {band_idx})")
                continue

    if not samples:
        raise ValueError("Could not read any valid data from the file. File may be completely corrupted.")

    samples = np.concatenate(samples)

    # Remove max values (65535 for uint16) before calculating percentiles
    max_value = 65535 if src.dtypes[0] == 'uint16' else np.iinfo(src.dtypes[0]).max
    valid_samples = samples[samples < max_value]

    if len(valid_samples) == 0:
        print("Warning: All sampled pixels are at max value. Using all samples.")
        valid_samples = samples
    else:
        removed_pct = (len(samples) - len(valid_samples)) / len(samples) * 100
        print(f"Removed {removed_pct:.2f}% of pixels at max value ({max_value})")

    min_samples, max_samples = np.percentile(valid_samples, percentiles)
    print(f"Global percentiles: {min_samples:.2f} ; {max_samples:.2f}")
    if corrupted_count > 0:
        print(f"Skipped {corrupted_count} corrupted tiles during sampling")

    return min_samples, max_samples



def process_orthomosaic(input_path, output_path, chunk_size=1024):
    """
    Process large orthomosaic by reading RGB bands and converting to uint8.

    Args:
        input_path: Path to input TIF file
        output_path: Path to output TIF file
        chunk_size: Size of chunks to process (pixels)
    """
    with rasterio.open(input_path) as src:
        # Get metadata
        profile = src.profile.copy()
        num_bands = src.count

        print(f"Input file has {num_bands} bands")
        print(f"Datatype: {src.dtypes[0]}")
        print(f"Dimensions: {src.width} x {src.height}")

        # Identify RGB bands (typically bands 1, 2, 3)
        rgb_bands = [1, 2, 3]

        # Works also with percentails (2,98)
        mi, mx = get_max_min_from_sampes(src, chunk_size, rgb_bands, percentiles=(0, 100))

        # Update profile for output
        profile.update(
            dtype=rasterio.uint8,
            count=3,
            compress='lzw',
            tiled=True,
            blockxsize=256,
            blockysize=256
        )

        # Process and write data
        with rasterio.open(output_path, 'w', **profile) as dst:
            print("Processing image in chunks...")
            corrupted_chunks = 0

            for i in range(0, src.height, chunk_size):
                rows = min(chunk_size, src.height - i)
                window = Window(0, i, src.width, rows)

                for band_idx, band_num in enumerate(rgb_bands, 1):
                    try:
                        # Read chunk
                        data = src.read(band_num, window=window)

                        # Convert using global percentiles
                        data_clipped = np.clip(data, mi, mx)
                        if mx > mi:
                            data_uint8 = ((data_clipped - mi) / (mx - mi) * 255).astype(np.uint8)
                        else:
                            data_uint8 = np.zeros_like(data, dtype=np.uint8)

                        # Write chunk
                        dst.write(data_uint8, band_idx, window=window)

                    except Exception as e:
                        corrupted_chunks += 1
                        print(f"\nWarning: Failed to read chunk at row {i}, band {band_num}. Writing zeros.")
                        # Write zeros for corrupted chunk
                        zero_data = np.zeros((rows, src.width), dtype=np.uint8)
                        dst.write(zero_data, band_idx, window=window)

                progress = ((i + rows) / src.height) * 100
                print(f"Progress: {progress:.1f}%", end='\r')

            print(f"\nProcessing complete!")
            if corrupted_chunks > 0:
                print(f"Warning: {corrupted_chunks} corrupted chunks were replaced with zeros")



if __name__ == "__main__":
    input_tif = ORTHO_IMG_DIR / "20250827_Bjørnkjærvej_TestFlight_2_mid.tif"   # path to your input
    output_tif = ORTHO_IMG_DIR / "RGB2_20250827_Bjørnkjærvej_TestFlight_2_mid.tif"
    process_orthomosaic(input_tif, output_tif, chunk_size=1024)
    print(f"Output saved to: {output_tif}")

