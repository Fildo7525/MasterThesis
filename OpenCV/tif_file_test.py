from numpy import uint16
import numpy as np
import rasterio
from rasterio.enums import Resampling
from create_indexes import Bands, Indices, compute_index
import cv2
from pathlib import Path
from typing import List

def read_multiband_tiff(path):
    with rasterio.open(path) as src:
        band_count = src.count
        data = src.read()  # shape: (bands, rows, cols)
        profile = src.profile.copy()
    return data, band_count, profile


def to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)

    min_val = arr.min()
    max_val = arr.max()

    if max_val == min_val:
        print("[ERROR]: While casting array to uint8 the min and max values of the array are the same.")
        return np.zeros(arr.shape, dtype=np.uint8)

    if max_val > min_val:
        arr = (arr - min_val) / (max_val - min_val)
    else:
        arr = np.zeros_like(arr)

    return (arr * 255).astype(np.uint8)


def calculate_indices(bands):
    results = []

    for index in Indices:
        band_name = index.name
        vegetation_index = compute_index(band_name, bands).astype(np.float32)

        img_vegetation_index = to_uint8(vegetation_index)
        print(f"{band_name:10}: min: {img_vegetation_index.min():20f}, max: {img_vegetation_index.max():20f}")

        cv2.imwrite(f"./images/{band_name}.png", img_vegetation_index)
        # cv2.imshow(band_name, vegetation_index)
        # cv2.waitKey(0)
        # cv2.destroyWindow(band_name)

        results.append(vegetation_index)

    arrs = np.stack(results, axis = 0)
    print(f"{arrs.shape}, {arrs.dtype}")
    return arrs


def get_index_names():
    names = []
    for band in Bands:
        names.append(band.name)

    for idx in Indices:
        names.append(idx.name)

    return names



def write_multiband_tiff(path, bands, profile):
    profile.update(
        count=bands.shape[0],
        dtype=bands.dtype,
        compress="lzw"
    )

    with rasterio.open(path, "w", **profile) as dst:
        names = get_index_names()
        print(f"Names length: {len(names)}")
        for i in range(bands.shape[0]):
            dst.write(bands[i], i + 1)
            dst.set_band_description(i+1, names[i])


def reference_band_indices(bands: List[Bands | Indices]) -> List[int]:
    return [b - 1 if isinstance(b, Bands) else 7 + b.value - 1 for b in bands]


def export2png(filename: Path,
               all_bands: np.ndarray,
               bands: List[Bands | Indices]) -> np.ndarray:

    out_mat = []
    out_bands = reference_band_indices(bands)

    for b in out_bands:
        band = all_bands[b, :, :].astype(np.float32)

        print(
            f"Adding band {b} to PNG export. "
            f"Min: {band.min()}, Max: {band.max()}"
        )

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


    print(f"Exporting PNG with shape: {img.shape}, dtype: {img.dtype}")

    cv2.imwrite(str(filename), img)
    cv2.imshow(str(filename), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return img


def main(input_tiff: Path, output_tiff: Path):
    bands, band_count, profile = read_multiband_tiff(input_tiff)

    print(f"bands: {bands.shape}, {bands.dtype}")
    indices = calculate_indices(bands)

    nbands = np.zeros_like(bands)
    for i in range(7):
        nbands[i,:,:] = to_uint8(bands[i,:,:])
        cv2.imwrite(f"./images/{Bands(i+1).name}.png", nbands[i,:,:])
        print(f"{Bands(i+1).name:20}: min: {nbands[i,:,:].min()}, max: {nbands[i,:,:].max()}")

    print(f"nbands: {nbands.shape}")
    all_bands = np.concatenate([bands[:7,:,:], indices], axis=0)

    export2png(Path("./tile_2_5_NRN.png"), all_bands, [Indices.NGRDI, Indices.RVI, Bands.NIR])

    indices = reference_band_indices([Indices.NGRDI, Indices.RVI, Bands.NIR])
    write_multiband_tiff(Path("./tile_2_5_NRN.tif"), all_bands[indices, :, :], profile)

    write_multiband_tiff(output_tiff, all_bands, profile)


if __name__ == "__main__":
    inp = Path("/home/fildo/SDU/MasterThesis/Orthomosaics/example_tiles/tile_2_5.tif")
    out = Path( "/home/fildo/SDU/MasterThesis/OpenCV/images/tile_2_5_b32.tif")
    main(inp, out)

