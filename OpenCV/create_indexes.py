import rasterio
import numpy as np
from enum import IntEnum
from pathlib import Path
import cv2


class Bands(IntEnum):
    RED = 1
    GREEN = 2
    BLUE = 3
    EXTEND_GREEN = 4
    EXTEND_RED = 5
    REDEDGE = 6
    NIR = 7


class Indices(IntEnum):
    NDVI = 1
    SVI = 2
    NGRDI = 3
    # Used to enhance the detection and analysis of green vegetation
    GNDVI = 4
    RVI = 5

    # Indicate the relationship between the absorbed radiant energy by vegetation and the reflectance in red,
    # green, and near-infrared bands, useful for monitoring crop biomass
    TVI = 6  # Triangle Vegetation Index

    NDRE = 7
    CVI = 8
    CIG = 9
    CIRE = 10
    DVI = 11
    SAVI = 12
    OSAVI = 13
    MSAVI2 = 14
    MGRVI = 15  # Modified Green-Red Vegetation Index (https://doi.org/10.1016/j.jag.2019.01.001)
    NGRVI = 16  # New Green-Red Vegetation Index (https://doi.org/10.1016/j.jag.2019.01.001)

    # Reflects the color characteristics of vegetation, can be used to identify vegetation types and estimate biomass
    CIVE = 17  # Color Index of Vegetation Extraction (https://doi.org/10.1016/j.rse.2008.06.006)

    # Reduces the influence of atmospheric and soil noise, providing a stable response to the vegetation condition in the measured area
    EVI = 18  # Enhanced Vegetation Index (https://doi.org/10.1016/S0034-4257(96)00066-3)

    # Used for detecting vegetation
    EXG = 19  # Excess Green Index

    # Can effectively distinguish green vegetation from non-vegetated areas in complex backgrounds
    EXGR = 20  # Excess Green-Red Index

    # Used for detecting non-vegetated areas
    EXR = 21  # Excess Red Index

    # Enhances vegetation signals while reducing atmospheric interference and soil background noise
    MSVI = 22  # Modified Soil-Adjusted Vegetation Index

    # Identify vegetated areas and reflect their health status
    NGBDI = 23  # Normalized Green-Blue Difference Index

    # Assess chlorophyll content and photosynthetic capacity of plants
    NPCI = 24  # Normalized Pigment Chlorophyll Index

    # Evaluate the leaf area index and biomass vegetation index
    RTVICore = 25  # Red-Edge Triangular Vegetation Index Core

    # Source Address Validation Improvement
    SAVI2 = 26  # Soil-Adjusted Vegetation Index 2

    # Monitor vegetation health, detect plant physiological stress, and analyze crop yield
    SIPI = 27  # Structure Insensitive Pigment Index

    # Common vegetation index for assessing vegetation quantity
    SR = 28  # Simple Ratio Index

    # Evaluate chlorophyll content and plant health
    TCARI = 29  # Transformed Chlorophyll Absorption in Reflectance Index

    # Reduce the impact of atmospheric conditions on vegetation index calculations
    VARI = 30  # Visible Atmospherically Resistant Index

    # Utilize differences in the visible spectrum to assess vegetation health and coverage
    VDVI = 31  # Visible Difference Vegetation Index

    # Vegetation index based on RGB channel information, used to estimate grassland vegetation coverage
    VEG = 32  # Vegetation Index


# -------- Theoretical (min, max) ranges for scaling to uint16 --------
# These define the expected output range of each index formula.
# Values are clipped to this range before scaling to 0–65535.
INDEX_RANGES = {
    "NDVI":      (-1.0,  1.0),
    "SVI":       (-1.0,  1.0),
    "NGRDI":     (-1.0,  1.0),
    "GNDVI":     (-1.0,  1.0),
    "RVI":       ( 0.0, 30.0),
    "TVI":       ( 0.0,  1.0),
    "NDRE":      (-1.0,  1.0),
    "CVI":       ( 0.0, 30.0),
    "CIG":       ( 0.0, 15.0),
    "CIRE":      ( 0.0, 15.0),
    "DVI":       (-1.0,  1.0),
    "SAVI":      (-1.0,  1.0),
    "OSAVI":     (-1.0,  1.0),
    "MSAVI2":    (-1.0,  1.0),
    "MGRVI":     (-1.0,  1.0),
    "NGRVI":     ( 0.0, 100.0),
    "CIVE":      (18.0, 21.0),
    "EVI":       (-1.0,  2.5),
    "EXG":       (-2.0,  2.0),
    "EXGR":      (-2.0,  2.0),
    "EXR":       (-1.0,  2.0),
    "MSVI":      (-10.0, 10.0),
    "NGBDI":     (-1.0,  1.0),
    "NPCI":      (-1.0,  1.0),
    "RTVICore":  (-100.0, 100.0),
    "SAVI2":     (-1.0,  1.0),
    "SIPI":      ( 0.0,  2.0),
    "SR":        ( 0.0, 30.0),
    "TCARI":     (-1.0,  3.0),
    "VARI":      (-1.0,  1.0),
    "VDVI":      (-1.0,  1.0),
    "VEG":       ( 0.0, 10.0),
}

UINT16_MAX = 65535


def scale_to_uint16(arr: np.ndarray, index_name: str) -> np.ndarray:
    """
    Clip a float index array to its theoretical range and scale linearly to uint16 (0–65535).

    The original float value can be recovered with:
        float_value = uint16_value / 65535 * (vmax - vmin) + vmin
    """
    vmin, vmax = INDEX_RANGES.get(index_name, (-1.0, 1.0))
    arr = np.clip(arr, vmin, vmax)
    arr = (arr - vmin) / (vmax - vmin)  # normalize to [0, 1]
    arr = (arr * UINT16_MAX).astype(np.uint16)
    return arr


# -------- Index Formulas --------
def compute_index(name: str, bands: list[np.ndarray]) -> np.ndarray:
    """
    Compute a vegetation index from a list of band arrays.

    Bands are expected to be normalized to [0, 1] (i.e. divided by 65535
    prior to calling this function). Returns a float32 array.
    """
    R   = bands[Bands.EXTEND_RED - 1]
    G   = bands[Bands.EXTEND_GREEN - 1]
    B   = bands[Bands.BLUE - 1]
    RE  = bands[Bands.REDEDGE - 1]
    NIR = bands[Bands.NIR - 1]

    eps    = 1e-5
    lmbda  = 0.667
    L      = 0.5

    formulas = {
        "CIG":      lambda: (NIR / (G + eps)) - 1,
        "CIRE":     lambda: (NIR / (RE + eps)) - 1,
        "CIVE":     lambda: 0.441 * B - 0.881 * G + 0.385 * R + 18.78745,
        "CVI":      lambda: (NIR * R) / (G ** 2 + eps),
        "DVI":      lambda: NIR - R,
        "EVI":      lambda: 2.5 * (NIR - R) / (NIR + 6 * R - 7.5 * B + 1 + eps),
        "EXG":      lambda: 2 * G - R - B,
        "EXGR":     lambda: 2 * G - 2.4 * R,
        "EXR":      lambda: 1.4 * R - B,
        "GNDVI":    lambda: (NIR - G) / (NIR + G + eps),
        "MGRVI":    lambda: (G ** 2 - R ** 2) / (G ** 2 + R ** 2 + eps),
        "MSAVI2":   lambda: 0.5 * (2 * NIR + 1 - np.sqrt((2 * NIR + 1) ** 2 - 8 * (NIR - R))),
        "MSVI":     lambda: ((NIR - R) - 1) / (np.sqrt(NIR / (R + eps) + 1) + eps),
        "NDRE":     lambda: (NIR - RE) / (NIR + RE + eps),
        "NDVI":     lambda: (NIR - R) / (NIR + R + eps),
        "NGBDI":    lambda: (G - B) / (G + B + eps),
        "NGRDI":    lambda: (G - R) / (G + R + eps),
        "NGRVI":    lambda: (G ** 2 + R ** 2) / (G ** 2 - R ** 2 + eps),
        "NPCI":     lambda: (R - B) / (R + B + eps),
        "OSAVI":    lambda: (NIR - R) / (NIR + R + 0.16),
        "RTVICore": lambda: 100 * (NIR - RE) - 10 * (NIR - G),
        "RVI":      lambda: NIR / (R + eps),
        "SAVI":     lambda: (NIR - R) / (NIR + R + L) * (1 + L),
        "SAVI2":    lambda: (NIR - R) / (NIR + R + 0.5) * 1.5,
        "SIPI":     lambda: (NIR - B) / (NIR - R + eps),
        "SR":       lambda: NIR / (R + eps),
        "SVI":      lambda: (NIR - G) / (NIR + G + eps),
        "TCARI":    lambda: 3 * ((RE - R) - 0.2 * (RE - G) * (RE / (R + eps))),
        "TVI":      lambda: np.sqrt(np.abs((NIR - R) / (NIR + R + eps) + 0.5)),
        "VARI":     lambda: (G - R) / (G + R - B + eps),
        "VDVI":     lambda: (2 * G - (R + B)) / (2 * G + (R + B) + eps),
        "VEG":      lambda: G / (R ** lmbda * B ** (1 - lmbda) + eps),
    }

    if name not in formulas:
        raise ValueError(f"Index '{name}' not implemented")

    return formulas[name]().astype(np.float32)


# -------- Processing --------
def read_normalized_bands(src: rasterio.DatasetReader) -> list[np.ndarray]:
    """
    Read all bands from an open rasterio dataset and normalize to [0, 1].
    Supports uint8 (max 255) and uint16 (max 65535) source data.
    """
    dtype = src.dtypes[0]
    if np.dtype(dtype) == np.uint8:
        scale = 255.0
    else:
        # Default to uint16 range; adjust if your data uses a different bit depth
        scale = float(UINT16_MAX)

    return [src.read(i).astype(np.float32) / scale for i in range(1, src.count + 1)]


def calculate_all_indices(input_path, output_path, indices):
    """
    Compute every index in *indices* for the given input raster and write
    each result as a single-band uint16 GeoTIFF.

    Output files are organized as:
        <output_path>/<INDEX_NAME>/<input_stem>_<index_name>.tif
    """
    input_path  = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    with rasterio.open(input_path) as src:
        meta  = src.profile.copy()
        meta.update(dtype=rasterio.uint16, count=len(indices))
        # meta.update(nodata=0)

        out_path = output_path / f"{input_path.stem}_all_indices.tif"
        with rasterio.open(out_path, "w", **meta) as dst:
            bands = read_normalized_bands(src)

            for index in indices:
                float_index  = compute_index(index.name, bands)
                uint16_index = scale_to_uint16(float_index, index.name)

                dst.write(uint16_index, int(index))
                cv2.imwrite(str(output_path / f"{input_path.stem}_{index.name.lower()}.png"), uint16_index)
                dst.set_band_description(int(index), index.name)


def caluclate_index(band, index: Indices):
    """
    Compute a single vegetation index from a single band array.

    This is a simplified interface for testing and visualization purposes.
    The band array should already be normalized to [0, 1].
    """
    # For testing, we can just return the input band as a placeholder
    return band


def calculate_index(input_path, output_path, index: Indices):
    """
    Compute a single vegetation index and write the result as a uint16 GeoTIFF.

    Returns the output Path on success, or None if the index is unknown.
    """
    if index.name not in Indices.__members__:
        return None

    input_path  = Path(input_path)
    output_path = Path(output_path)

    index_dir = output_path / index.name
    index_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(input_path) as src:
        meta  = src.profile.copy()
        meta.update(dtype=rasterio.uint16, count=1)

        bands = read_normalized_bands(src)
        float_index = compute_index(index.name, bands)
        uint16_index = scale_to_uint16(float_index, index.name)

        out_path = index_dir / f"{input_path.stem}_{index.name.lower()}.tif"
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(uint16_index, 1)

    return out_path


if __name__ == "__main__":
    DIR_PATH = Path.home() / "SDU/MasterThesis/Orthomosaics/example_tiles_small"
    input_tif = DIR_PATH / "tile_2_5.tif"
    output_dir = Path.cwd() / "indices_output"

    calculate_all_indices(input_tif, output_dir, Indices)
