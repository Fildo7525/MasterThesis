import matplotlib
from numpy import uint16
import numpy as np
import rasterio
from rasterio.enums import Resampling
from create_indexes import Bands, Indices, compute_index
import cv2
from pathlib import Path
from typing import List
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

DBG = False

def read_multiband_tiff(path: Path) -> tuple[np.ndarray, int, dict]:
    """ Read all the bands from the TIFF file.

    Args:
        path (pathlib.Path): Path to the TIFF file.

    Returns:
        Tuple[np.ndarray, int, dict]: A tuple containing:
            - A NumPy array of shape (bands, rows, cols) with the band data.
            - An integer representing the number of bands.
            - A dictionary with the raster profile metadata.
    """
    with rasterio.open(path, "r") as src:
        band_count = src.count
        data = src.read()  # shape: (bands, rows, cols)
        profile = src.profile.copy()
    return data, band_count, profile


def calculate_indices(bands: np.ndarray) -> np.ndarray:
    """ Calculate vegetation indices from the given bands.

    Args:
        bands (np.ndarray): A NumPy array of shape (bands, rows, cols) containing the band data.

    Returns:
        np.ndarray: A NumPy array of shape (num_indices, rows, cols) containing the calculated indices.
    """
    results = []

    get_line = lambda I: f"{I.name}: Min: {np.min(bands[I-1].flatten()):05.3f} Max: {np.max(bands[I-1].flatten()):05.3f}, type: {bands[I-1].dtype}, shape: {bands[I-1].shape}"
    if DBG:
        print(f"""
Bands:
{get_line(Bands.RED)}
{get_line(Bands.GREEN)}
{get_line(Bands.BLUE)}
{get_line(Bands.NIR)}
{get_line(Bands.REDEDGE)}
    """)
    for index in Indices:
        band_name = index.name
        vegetation_index = compute_index(band_name, bands)
        max_value = max(np.max(vegetation_index), 25000)

        if max_value is not None:
            if np.any(vegetation_index < 0):
                vegetation_index = (vegetation_index - vegetation_index.min()) / (vegetation_index.max() - vegetation_index.min()) * max_value

            elif np.all(vegetation_index <= 1):
                vegetation_index *= max_value

            elif np.all(vegetation_index <= 255):
                vegetation_index = (vegetation_index / 255) * max_value

        results.append(vegetation_index)

    arrs = np.stack(results, axis = 0)
    return arrs


def get_index_names():
    names = []
    for band in Bands:
        names.append(band.name)

    for idx in Indices:
        names.append(idx.name)

    return names


def write_multiband_tiff(path: Path, bands: np.ndarray, profile: dict, indices: List[Bands | Indices]):
    """ Write multiple bands to a TIFF file with updated profile and band descriptions.

    Args:
        path (pathlib.Path): Path to the output TIFF file.
        bands (np.ndarray): A NumPy array of shape (bands, rows, cols) containing the band data to write.
        profile (dict): Profile metadata for the TIFF file.
        indices: List[Bands | Indices]: List of band and index enums to set band descriptions.
    """
    profile.update(
        count=bands.shape[0],
        dtype=bands.dtype,
        compress="lzw"
    )

    names = []
    if len(indices) == 0:
        names = get_index_names()
    else:
        names = [i.name for i in indices]

    with rasterio.open(path, "w", **profile) as dst:
        # print(f"Names length: {len(names)}")
        for i in range(bands.shape[0]):
            dst.write(bands[i], i + 1)
            dst.set_band_description(i+1, names[i])


def reference_band_indices(bands: List[Bands | Indices | int]) -> List[int]:
    values = []
    for b in bands:
        if isinstance(b, Bands):
            values.append(b.value - 1)
        elif isinstance(b, Indices):
            values.append(7 + b.value - 1)
        elif isinstance(b, int):
            values.append(b)
        else:
            raise ValueError(f"Invalid band/index type: {type(b)}")

    return values


def export2png(filename: Path,
               all_bands: np.ndarray,
               bands: List[Bands | Indices | int] | None = None) -> np.ndarray:

    out_mat = []
    out_bands = reference_band_indices(bands) if bands is not None else list(range(all_bands.shape[0]))
    _, axs = plt.subplots(len(out_bands), 1, sharex=True)

    for b, ax in zip(out_bands, axs):
        band = all_bands[b, :, :].astype(np.float32)

        # if b in reference_band_indices([Indices.NGRDI, Indices.RVI]):
        #     print(f"Applying special scaling for index band {b}...")
        #     band = band.clip(-30, 30)  # Clip to [-1, 1]
        #     band = ((band + 30) / 60) * 65535

        if DBG:
            print(f"Band {b}: Min: {np.min(band):05.3f} Max: {np.max(band):05.3f}, type: {band.dtype}, shape: {band.shape}")

        # # Min–max normalization
        # min_v = band.min()
        # max_v = band.max()
        # if max_v > min_v:
        #     band = (band - min_v) / (max_v - min_v)
        # else:
        #     band = np.zeros_like(band)

        # band = (band * 255).astype(np.uint8)
        # # band = (np.round(np.sqrt(band))).astype(np.uint8)
        out_mat.append(band)

        if DBG:
            flat = band.flatten()
            ax.hist(flat, bins=100, color='blue', alpha=0.7)
            ax.set_title(f"Band {b}")

            print(f"Ouptut min: {np.min(band)}, max: {np.max(band)}, mean: {np.mean(band)}, std: {np.std(band)}, type: {band.dtype}, shape: {band.shape}")
            print("")

    out_mat.reverse()
    img = np.stack(out_mat, axis=0)      # (C, H, W)
    img = img.transpose(1, 2, 0)         # (H, W, C)

    if not filename.parent.exists():
        filename.parent.mkdir(parents=True, exist_ok=True)

    if filename.suffix.lower() != ".png":
        filename = filename.with_suffix(".png")

    if DBG:
        plt.savefig(filename.with_suffix(".histogram.png"))

    cv2.imwrite(str(filename), img)
    if DBG:
        print(f"Exporting {filename} with bands: {bands if bands is not None else 'all'}")
        cv2.imshow(str(filename), img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        plt.close()

    return img


def main(input_tiff: Path, output_tiff: Path):
    bands, _, profile = read_multiband_tiff(input_tiff)
    # print(f"Shape of bands: {bands.shape}, dtype: {bands.dtype}")

    indices = calculate_indices(bands)
    all_bands = np.concatenate([bands[:7,:,:], indices], axis=0)

    indices_to_export = [Indices.NGRDI, Indices.RVI, Bands.NIR]

    # png_file = output_tiff.parent / "pngs" / str(input_tiff.name).replace(".tif", ".png")
    # export2png(png_file, all_bands, indices_to_export)

    indices = reference_band_indices(indices_to_export)
    assert input_tiff != output_tiff, "Input and output paths are the same!"
    requested_bands = all_bands[indices, :, :]
    ranges= []
    for i in range(requested_bands.shape[0]):
        band = requested_bands[i, :, :]
        min_v, max_v= np.min(band), np.max(band)
        ranges.append((min_v, max_v))

    write_multiband_tiff(output_tiff, all_bands[indices, :, :], profile, indices_to_export)
    return ranges


def get_colour_ranges(inp: Path):
    def get_tile_colour_ranges(tile_path: Path):
        ranges = []
        bands, *_ = read_multiband_tiff(tile_path)
        for i in range(bands.shape[0]):
            band = bands[i, :, :]
            min_v, max_v= np.min(band), np.max(band)
            ranges.append((min_v, max_v))
        return ranges

    tiff_files = list(inp.glob("*.tif"))
    minmax_ranges = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(tqdm(executor.map(get_tile_colour_ranges, tiff_files), total=len(tiff_files), desc="TIFF"))
        for tile_path, ranges in zip(tiff_files, results):
            print(f"Tile: {tile_path.name}")
            for i, (min_v, max_v) in enumerate(ranges):
                if min_v < 0:
                    min_v = max(min_v, minmax_ranges.get(i, 0.0)[0])

                if max_v < 0:
                    max_v = max(max_v, minmax_ranges.get(i, 0.0)[1])

                print(f"  Band {i+1}: Min: {min_v}, Max: {max_v}")
                if minmax_ranges.get(i) is None:
                    minmax_ranges[i] = (min_v, max_v)
                else:
                    current_min,  current_max = minmax_ranges[i]
                    minmax_ranges[i] = (min(current_min, min_v), max(current_max, max_v))
            print("")

    print("Overall min-max ranges across all tiles:")
    for i, (min_v, max_v) in minmax_ranges.items():
        print(f"  Band {i+1}: Min: {min_v}, Max: {max_v}")
    return minmax_ranges


if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor
    inp = Path("/home/fildo/SDU/MasterThesis/Orthomosaics/example_tiles_big_2048/")
    out = Path("/home/fildo/SDU/MasterThesis/Orthomosaics/NRN_big_v2/")
    workers: int = 4

    if not out.exists():
        out.mkdir(parents=True, exist_ok=True)

    if DBG:
        # tile = inp / "tile_2_5.tif"
        # output_tile = out / "tile_2_5.tif"
        # main(tile, output_tile)
        # print(f"✅ Processed {tile} -> {output_tile}")

        get_colour_ranges(inp)

    else:
        tiff_files = list(inp.glob("*.tif"))
        # for file in tqdm(tiff_files, desc="TIFF"):
        #     input_tiff = inp / file.name
        #     output_tiff = out / file.name
        #     try:
        #         main(input_tiff, output_tiff)
        #     except Exception as e:
        #         pass

        def process_file(file):
            # print(f"Processing {file.name}...")
            input_tiff = inp / file.name
            output_tiff = out / file.name
            try:
                return main(input_tiff, output_tiff)
            except Exception as e:
                return [(0, 0)]

        minmax_ranges = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(tqdm(executor.map(process_file, tiff_files), total=len(tiff_files), desc="TIFF"))
            for tile_path, ranges in zip(tiff_files, results):
                # print(f"Tile: {tile_path.name}")
                for i, (min_v, max_v) in enumerate(ranges):
                    if min_v < 0:
                        min_v = max(min_v, minmax_ranges.get(i, 0.0)[0])

                    if max_v < 0:
                        max_v = max(max_v, minmax_ranges.get(i, 0.0)[1])

                    # print(f"  Band {i+1}: Min: {min_v}, Max: {max_v}")
                    if minmax_ranges.get(i) is None:
                        minmax_ranges[i] = (min_v, max_v)
                    else:
                        current_min,  current_max = minmax_ranges[i]
                        minmax_ranges[i] = (min(current_min, min_v), max(current_max, max_v))
                # print("")

        print("Overall min-max ranges across all tiles:")
        for i, (min_v, max_v) in minmax_ranges.items():
            print(f"  Band {i+1}: Min: {min_v}, Max: {max_v}")

