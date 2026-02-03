from numpy import uint16
import numpy as np
import rasterio
from rasterio.enums import Resampling
from create_indexes import Bands, Indices, compute_index
import cv2
from pathlib import Path
from typing import List
from tqdm import tqdm

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

    for index in Indices:
        band_name = index.name
        vegetation_index = compute_index(band_name, bands).astype(np.float32)
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

    for b in out_bands:
        band = all_bands[b, :, :].astype(np.uint16)

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
    img = np.stack(out_mat, axis=0)      # (C, H, W)
    img = img.transpose(1, 2, 0)         # (H, W, C)

    if not filename.parent.exists():
        filename.parent.mkdir(parents=True, exist_ok=True)

    if filename.suffix.lower() != ".png":
        filename = filename.with_suffix(".png")

    cv2.imwrite(str(filename), img)
    if DBG:
        cv2.imshow(str(filename), img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img


def main(input_tiff: Path, output_tiff: Path):
    bands, _, profile = read_multiband_tiff(input_tiff)
    # print(f"Shape of bands: {bands.shape}, dtype: {bands.dtype}")

    indices = calculate_indices(bands)
    all_bands = np.concatenate([bands[:7,:,:], indices], axis=0)

    png_file = input_tiff.parent.parent / "pngs" / str(input_tiff.name).replace(".tif", ".png")
    export2png(png_file, all_bands, [Indices.NGRDI, Indices.RVI, Bands.NIR])

    indices = reference_band_indices([Indices.NGRDI, Indices.RVI, Bands.NIR])
    assert input_tiff != output_tiff, "Input and output paths are the same!"
    write_multiband_tiff(output_tiff, all_bands[indices, :, :], profile, [Indices.NGRDI, Indices.RVI, Bands.NIR])


if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor, as_completed
    inp = Path("/home/fildo/SDU/MasterThesis/Orthomosaics/example_tiles")
    out = Path("/home/fildo/SDU/MasterThesis/Orthomosaics/NRN")
    workers: int = 4

    if not out.exists():
        out.mkdir(parents=True, exist_ok=True)

    if DBG:
        tile = inp / "tile_2_5.tif"
        output_tile = out / "tile_2_5.tif"
        main(tile, output_tile)
        print(f"✅ Processed {tile} -> {output_tile}")

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
            input_tiff = inp / file
            output_tiff = out / file
            try:
                main(input_tiff, output_tiff)
            except Exception as e:
                pass

        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(tqdm(executor.map(process_file, tiff_files), total=len(tiff_files), desc="TIFF"))

