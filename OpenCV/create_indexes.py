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
    GNDVI = 4
    RVI = 5
    TVI = 6
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


# -------- Index Formulas --------
def compute_index(name, bands):
    R = bands[Bands.EXTEND_RED - 1]
    G = bands[Bands.EXTEND_GREEN - 1]
    B = bands[Bands.BLUE - 1]
    RE = bands[Bands.REDEDGE - 1]
    NIR = bands[Bands.NIR - 1]

    eps = 1e-5
    L = 0.5

    formulas = {
        "NDVI": lambda: (NIR - R) / (NIR + R + eps),
        "SVI": lambda: (NIR - G) / (NIR + G + eps),
        "NGRDI": lambda: (G - R) / (G + R + eps),
        "GNDVI": lambda: (NIR - G) / (NIR + G + eps),
        "RVI": lambda: NIR / (R + eps),
        "TVI": lambda: np.sqrt(np.abs((NIR - R) / (NIR + R + eps) + 0.5)),
        "NDRE": lambda: (NIR - RE) / (NIR + RE + eps),
        "CVI": lambda: (NIR * R) / (G**2 + eps),
        "CIG": lambda: (NIR / (G + eps)) - 1,
        "CIRE": lambda: (NIR / (RE + eps)) - 1,
        "DVI": lambda: NIR - R,
        "SAVI": lambda: (NIR-R)*(1+L)/(NIR+R+L),
        "OSAVI": lambda: (NIR - R)/(NIR + R + 0.16),
        "MSAVI2": lambda: 0.5*(2*NIR+1-np.sqrt((2*NIR+1)**2-8*(NIR-R))),
        "MGRVI": lambda: (G**2 - R**2) / (G**2 + R**2 + eps),
        "NGRVI": lambda: (G**2 + R**2) / (G**2 - R**2 + eps),
    }

    if name not in formulas:
        raise ValueError(f"Index '{name}' not implemented")

    return formulas[name]()


# -------- Processing --------
def calculate_all_indices(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    os.makedirs(output_path, exist_ok=True)

    with rasterio.open(input_path) as src:
        meta = src.profile
        meta.update(dtype=rasterio.float32, count=1)

        for index in Indices:

            output_path_copy = output_path / str(index.name)
            os.makedirs(output_path_copy, exist_ok=True)

            bands = [src.read(i).astype(np.float32) for i in range(1, src.count + 1)]
            img_index = compute_index(index.name, bands)

            out_path = output_path_copy / f"{input_path.stem}_{index.name.lower()}.tif"
            with rasterio.open(out_path, "w", **meta) as dst:
                dst.write(img_index.astype(np.float32), 1)


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

        print(f"Calculating {index.name}...")

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

    calculate_all_indices(input_tif, output_dir)
