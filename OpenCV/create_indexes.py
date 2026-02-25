import cv2
import rasterio
import numpy as np
from enum import IntEnum
from pathlib import Path
import matplotlib.pyplot as plt
import os

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
    TVI = 6 # Triangle Vegetation Index

    NDRE = 7
    CVI = 8
    CIG = 9
    CIRE = 10
    DVI = 11
    SAVI = 12
    OSAVI = 13
    MSAVI2 = 14
    MGRVI = 15 # Modified Green-Red Vegetation Index (https://doi.org/10.1016/j.jag.2019.01.001)
    NGRVI = 16 # New Green-Red Vegetaion Index (https://doi.org/10.1016/j.jag.2019.01.001)

    #Reflects the color characteristics of vegetation, can be used to identify vegetation types and estimate biomass
    CIVE = 17 # Color Index of Vegetation Extraction (https://doi.org/10.1016/j.rse.2008.06.006)

    # Reduces the influence of atmospheric and soil noise, providing a stable response to the vegetation condition in the measured area
    EVI = 18  # Enhanced Vegetation Index (https://doi.org/10.1016/S0034-4257(96)00066-3)

    # Used for detecting vegetation
    EXG = 19  # Excess Green Index

    # Can effectively distinguish green vegetation from non-vegetated areas in complex backgrounds
    EXGR = 20 # Excess Green-Red Index

    # Used for detecting non-vegetated areas
    EXR = 21 # Excess Red Index

    # Enhances vegetation signals while reducing atmospheric interference and soil background noise
    MSRI = 22 # Modified Soil-Adjusted Vegetation Index

    # Identify vegetated areas and reflect their health status
    NGBDI = 23 # Normalized Green-Blue Difference Index

    # Assess chlorophyll content and photosynthetic capacity of plants
    NPCI = 24 # Normalized Pigment Chlorophyll Index

    # Evaluate the leaf area index and biomass vegetation index
    RTVICore = 25 # Red-Edge Triangular Vegetation Index Core

    # Source Address Validation Improvement
    SAVI2 = 26 # Source Address Validation Improvement

    # Monitor vegetation health, detect plant physiological stress, and analyze crop yield
    SIPI = 27 # Structure Insensitive Pigment Index

    #Common vegetation index for assessing vegetation quantity
    SR = 28 # Simple Ratio Index

    # Evaluate chlorophyll content and plant health
    TCARI = 29 # Transformed Chlorophyll Absorption in Reflectance Index

    # Reduce the impact of atmospheric conditions on vegetation index calculations
    VARI = 30 # Visible Atmospherically Resistant Index

    # Utilize differences in the visible spectrum to assess vegetation health and coverage
    VDVI = 31 # Visible Difference Vegetation Index

    # Vegetation index based on RGB channel information, used to estimate grassland vegetation coverage
    VEG = 32 # Vegetation Index


# -------- Index Formulas --------
def compute_index(name, bands):
    R = bands[Bands.EXTEND_RED - 1]
    G = bands[Bands.EXTEND_GREEN - 1]
    B = bands[Bands.BLUE - 1]
    RE = bands[Bands.REDEDGE - 1]
    NIR = bands[Bands.NIR - 1]

    eps = 1e-5
    lmbda = 0.667
    L = 0.5

    formulas = {
        "CIG": lambda: (NIR / (G + eps)) - 1,
        "CIRE": lambda: (NIR / (RE + eps)) - 1,
        "CIVE": lambda: 0.441*B - 0.881*G + 0.385*R + 18.78745,
        "CVI": lambda: (NIR * R) / (G**2 + eps),
        "DVI": lambda: NIR - R,
        "EVI": lambda: 2.5 * (NIR - R), # / (NIR + 6 * R - 7.5 * B + 1 + eps),
        "EXG": lambda: 2 * G - R - B,
        "EXGR": lambda: 2 * G - 2.4 * R,
        "EXR": lambda: 1.4 * R - B,
        "GNDVI": lambda: (NIR - G) / (NIR + G + eps),
        "MGRVI": lambda: (G**2 - R**2) / (G**2 + R**2 + eps),
        "MSAVI2": lambda: 0.5*(2*NIR+1-np.sqrt((2*NIR+1)**2-8*(NIR-R))),
        "MSRI": lambda: ((NIR - R) - 1) / (np.sqrt(NIR / (R + eps) + 1) + eps),
        "NDRE": lambda: (NIR - RE) / (NIR + RE + eps),
        "NDVI": lambda: (NIR - R) / (NIR + R + eps),
        "NGBDI": lambda: (G - B) / (G + B + eps),
        "NGRDI": lambda: (G - R) / (G + R + eps),
        "NGRVI": lambda: (G**2 + R**2) / (G**2 - R**2 + eps),
        "NPCI": lambda: (R - B) / (R + B + eps),
        "OSAVI": lambda: (NIR - R)/(NIR + R + 0.16),
        "RTVICore": lambda: 100 * (NIR - RE) - 10 * (NIR - G),
        "RVI": lambda: NIR / (R + eps),
        "SAVI": lambda: (NIR-R)*(1+L)/(NIR+R+L),
        "SAVI2": lambda: (NIR - R) / (NIR + R + 0.5) * 1.5,
        "SIPI": lambda: (NIR - B) / (NIR - R + eps),
        "SR": lambda: NIR / (R + eps),
        "SVI": lambda: (NIR - G) / (NIR + G + eps),
        "TCARI": lambda: 3 * ((RE - R) - 0.2 * (RE - G) * (RE / (R + eps))),
        "TVI": lambda: np.sqrt(np.abs((NIR - R) / (NIR + R + eps) + 0.5)),
        "VARI": lambda: (G - R) / (G + R - B + eps),
        "VDVI": lambda: (2*G - (R + B)) / (2*G + (R + B) + eps),
        "VEG": lambda: G / (R**lmbda * B**(1 - lmbda) + eps),
    }

    if name not in formulas:
        raise ValueError(f"Index '{name}' not implemented")

    return formulas[name]()


# -------- Processing --------
def calculate_all_indices(input_path, output_path, indices):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    os.makedirs(output_path, exist_ok=True)

    with rasterio.open(input_path) as src:
        meta = src.profile
        meta.update(dtype=rasterio.float32, count=1)

        for index in indices:
            output_path_copy = output_path / str(index.name)
            os.makedirs(output_path_copy, exist_ok=True)

            bands = [src.read(i).astype(np.float32) for i in range(1, src.count + 1)]
            img_index = compute_index(index.name, bands)

            out_path = output_path_copy / f"{input_path.stem}_{index.name.lower()}.tif"
            cv2.imwrite(str(out_path), img_index.astype(np.float32))


def calculate_index(input_path, output_path, index: Indices):

    if index.name not in Indices.__members__:
        return None

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    os.makedirs(output_path, exist_ok=True)

    with rasterio.open(input_path) as src:
        meta = src.profile
        meta.update(dtype=rasterio.float32, count=1)

        output_path_copy = output_path / str(index)
        os.makedirs(output_path_copy, exist_ok=True)

        bands = [src.read(i).astype(np.float32) for i in range(1, src.count + 1)]
        img_index = compute_index(index.name, bands)

        out_path = output_path_copy / f"{input_path.stem}_{index.name.lower()}.tif"
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(img_index.astype(np.float32), 1)



if __name__ == "__main__":
    DIR_PATH = Path("/home/samuel/Documents/code/Orthomosaics/20250827_Bjørnkjærvej_TestFlight_2_small_tiles").expanduser()
    input_tif = DIR_PATH / "tile_1_1.tif"
    output_dir = DIR_PATH / "indices_output"

    calculate_all_indices(input_tif, output_dir, Indices)
