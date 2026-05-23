import os
from pathlib import Path
import cv2
import numpy as np
import rasterio
from dataclasses import dataclass
from typing import Tuple
from tqdm import tqdm
from enum import IntEnum

from .image_splitter import split_geotiff

from qgis.core import (
    Qgis,
    QgsMessageLog,
)

# ---------------------------------------------------------------------
# Theoretical index ranges for scaling
# ---------------------------------------------------------------------

@dataclass
class ThreeBandInputPaths:
    band1: Path
    band2: Path
    band3: Path



#!/usr/bin/env python3

import rasterio
import numpy as np
from enum import IntEnum
from pathlib import Path
import os
import tqdm
# ---------------------------------------------------------------------
# Band and Index Enums
# ---------------------------------------------------------------------


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
    MGRVI = 15
    NGRVI = 16
    CIVE = 17
    EVI = 18
    EXG = 19
    EXGR = 20
    EXR = 21
    MSVI = 22
    NGBDI = 23
    NPCI = 24
    RTVICore = 25
    SAVI2 = 26
    SIPI = 27
    SR = 28
    TCARI = 29
    VARI = 30
    VDVI = 31
    VEG = 32
    NGRDI_EXTENDED = 33
    NGRDI_REAL = 34


# ---------------------------------------------------------------------
# Default band ranges (fallback if global stats not provided)
# ---------------------------------------------------------------------

BAND_MAX_MIN_VALUES = {
    Bands.RED: (0, 65535),
    Bands.GREEN: (0, 65535),
    Bands.BLUE: (0, 65535),
    Bands.EXTEND_GREEN: (0, 400),
    Bands.EXTEND_RED: (0, 400),
    Bands.REDEDGE: (0, 400),
    Bands.NIR: (0, 65535),
}


# ---------------------------------------------------------------------
# Theoretical index ranges for scaling
# ---------------------------------------------------------------------

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
    "NGRDI_EXTENDED": (-1.0, 1.0),
    "NGRDI_REAL": (-1.0, 1.0),
}

UINT16_MAX = 65535



def compute_global_band_stats(tile_dir: Path, fill_threshold: int = 100) -> dict:
    """
    Compute global min/max for all 7 bands across all tiles.
    Returns a dict keyed by Bands enum, value = (min, max).
    """
    # Band index in the .tif file (1-based) → Bands enum
    band_map = {
        1: Bands.RED,
        2: Bands.GREEN,
        3: Bands.BLUE,
        4: Bands.EXTEND_GREEN,
        5: Bands.EXTEND_RED,
        6: Bands.REDEDGE,
        7: Bands.NIR,
    }

    accum = {b: {"mins": [], "maxs": []} for b in band_map.values()}

    tile_paths = sorted(tile_dir.glob("*.tif"))
    if not tile_paths:
        raise FileNotFoundError(f"No .tif files found in {tile_dir}")

    for tile_path in tqdm.tqdm(tile_paths, desc="Computing global band stats"):
        with rasterio.open(tile_path) as src:
            for band_idx, band_enum in band_map.items():
                if band_idx > src.count:
                    continue
                data = src.read(band_idx).astype(np.float32)
                nodata = src.nodata

                if nodata is not None:
                    valid = data[data != nodata]
                else:
                    valid = data.ravel()

                valid = valid[valid >= fill_threshold]
                if valid.size == 0:
                    continue

                accum[band_enum]["mins"].append(float(np.min(valid)))
                accum[band_enum]["maxs"].append(float(np.max(valid)))

    result = {}
    for band_enum, vals in accum.items():
        if vals["mins"]:
            result[band_enum] = (
                float(np.mean(vals["mins"])),
                float(np.mean(vals["maxs"])),
            )
        else:
            # Fall back to defaults
            result[band_enum] = BAND_MAX_MIN_VALUES[band_enum]

    return result

# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def normalize_band(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Normalize band to [0,1] with clamping."""
    arr = (arr - vmin) / (vmax - vmin + 1e-6)
    return np.clip(arr, 0.0, 1.0)


def scale_to_uint16(arr: np.ndarray, index_name: str) -> np.ndarray:
    """Scale float index to uint16 using theoretical range."""
    vmin, vmax = INDEX_RANGES.get(index_name, (-1.0, 1.0))
    arr = np.clip(arr, vmin, vmax)
    arr = (arr - vmin) / (vmax - vmin + 1e-6)
    return (arr * UINT16_MAX).astype(np.uint16)

def scale_to_uint8(arr: np.ndarray, index_name: str,
                   nodata_out: int = 0) -> np.ndarray:
    vmin, vmax = INDEX_RANGES.get(index_name, (-1.0, 1.0))
    arr = np.clip(arr, vmin, vmax)
    arr = (arr - vmin) / (vmax - vmin + 1e-6)
    out = (arr * 255).astype(np.float32)

    # NaN → nodata_out (0 = pure black, unambiguous background)
    nan_mask = np.isnan(arr)
    out[nan_mask] = nodata_out
    return out.astype(np.uint8)


# ---------------------------------------------------------------------
# Updated compute_index — takes effective_ranges directly (no copy inside)
# ---------------------------------------------------------------------

def compute_index(name: str, bands: list, min_max_values: dict = None,
                  alpha: np.ndarray = None) -> np.ndarray:
    """
    alpha: if provided (band 8), pixels where alpha==0 are set to NaN
           so they get a fixed nodata value (0) in the output, not a
           spurious index value.
    """
    R   = bands[Bands.RED   - 1].astype(np.float32)
    G   = bands[Bands.GREEN - 1].astype(np.float32)
    B   = bands[Bands.BLUE         - 1].astype(np.float32)
    RE  = bands[Bands.REDEDGE      - 1].astype(np.float32)
    NIR = bands[Bands.NIR          - 1].astype(np.float32)
    ER = bands[Bands.EXTEND_RED    - 1].astype(np.float32)
    EG = bands[Bands.EXTEND_GREEN  - 1].astype(np.float32)

    ranges = BAND_MAX_MIN_VALUES.copy()
    if min_max_values:
        ranges.update(min_max_values)
        R   = normalize_band(R,   *ranges[Bands.RED])
        G   = normalize_band(G,   *ranges[Bands.GREEN])
        ER  = normalize_band(ER,  *ranges[Bands.EXTEND_RED])
        EG  = normalize_band(EG,  *ranges[Bands.EXTEND_GREEN])
        B   = normalize_band(B,   *ranges[Bands.BLUE])
        RE  = normalize_band(RE,  *ranges[Bands.REDEDGE])
        NIR = normalize_band(NIR, *ranges[Bands.NIR])

    eps = 1e-5
    L   = 0.5

    formulas = {
        "NDVI":   lambda: (NIR - R) / (NIR + R + eps),
        "NGRDI_EXTENDED":  lambda: (EG - ER)   / (EG + ER + eps),
        "NGRDI_REAL":  lambda: (G - R)   / (G + R + eps),
        "GNDVI":  lambda: (NIR - G) / (NIR + G + eps),
        "NDRE":   lambda: (NIR - RE) / (NIR + RE + eps),
        "RVI":    lambda: NIR / (R + eps),
        "DVI":    lambda: NIR - R,
        "SAVI":   lambda: (NIR - R) / (NIR + R + L) * (1 + L),
        "OSAVI":  lambda: (NIR - R) / (NIR + R + 0.16),
        "MSAVI2": lambda: 0.5 * (2 * NIR + 1 - np.sqrt(
                      np.clip((2 * NIR + 1)**2 - 8 * (NIR - R), 0, None))),
        "EXG":    lambda: 2 * G - R - B,
        "VARI":   lambda: (G - R) / (G + R - B + eps),
        "EXGR":   lambda: 2 * G - 2.4 * R,
        "MGRVI":  lambda: (G*G - R*R) / (G*G + R*R + eps),
        "NGRVI":  lambda: (G*G + R*R) / (G*G - R*R + eps),
    }

    if name not in formulas:
        raise ValueError(f"Index '{name}' not implemented")

    result = formulas[name]().astype(np.float32)

    # Zero out nodata pixels so they don't appear as random grey
    if alpha is not None:
        result[alpha == 0] = np.nan

    return result

# ---------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------

def calculate_all_indices(input_path, output_path, indices,
                           do_gloabal_norm: bool=False):
    input_path  = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Compute ONCE, outside the tile loop
    if do_gloabal_norm:
        global_stats = compute_global_band_stats(input_path)
    else:
        global_stats = None
    print(global_stats)

    image_tiles = sorted(input_path.glob("*.tif"))
    for image_tile in tqdm.tqdm(image_tiles, desc="Processing tiles"):
        with rasterio.open(image_tile) as src:
            bands = [src.read(i).astype(np.float32) for i in range(1, src.count + 1)]

            # Extract alpha if band 8 exists
            alpha = src.read(8) if src.count >= 8 else None

            for index in indices:
                index_name = index.name
                index_dir  = output_path
                index_dir.mkdir(exist_ok=True)

                img_float = compute_index(index_name, bands, global_stats, alpha=alpha)
                img_uint8 = scale_to_uint8(img_float, index_name, nodata_out=0)

                meta = {
                    "driver":    "GTiff",
                    "height":    src.height,
                    "width":     src.width,
                    "count":     1,
                    "dtype":     rasterio.uint8,
                    "crs":       src.crs,
                    "transform": src.transform,
                    "nodata":    0,        # ← tell QGIS what nodata is
                    "compress":  "lzw",
                }

                out_path = index_dir / f"{image_tile.stem}.tif"
                with rasterio.open(out_path, "w", **meta) as dst:
                    dst.write(img_uint8, 1)

# ---------------------------------------------------------------------
# Single index helper
# ---------------------------------------------------------------------


def normalize_tile_global_blend(img: np.ndarray, global_min: float, global_max: float, blend: float = 0.1) -> np.ndarray:
        """
        Normalize a tile using pre-computed global min/max, but blend slightly with
        the tile's actual min/max to reduce visible seams.

        blend: 0 = fully global min/max
            1 = fully per-tile min/max
        """
        img = img.astype(np.uint16)

        tile_min = np.min(img)
        tile_max = np.max(img)

        # Blend per-tile min/max with global
        scaled_min = (1 - blend) * global_min + blend * tile_min
        scaled_max = (1 - blend) * global_max + blend * tile_max

        if scaled_max == scaled_min:
            return np.zeros_like(img, dtype=np.uint8)

        scaled = (img - scaled_min) / (scaled_max - scaled_min)
        scaled = np.clip(scaled, 0, 1) * 255
        return scaled.astype(np.uint8)

def normalize_tile(img):
        img = img.astype(np.float32)
        return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def separate_band(input_path: Path, output_path: Path, band: Bands,
                global_min: float | None = None,
                global_max: float | None = None):

    os.makedirs(output_path, exist_ok=True)
    name = Path(input_path).stem

    with rasterio.open(input_path) as src:
        img = src.read(band.value).astype(np.float32)

        # Use global stats if provided, otherwise fall back to local (with a warning)
        if global_min is not None and global_max is not None:
            img = normalize_tile_global_blend(img, global_min, global_max)
        else:
            img = normalize_tile(img)


        profile = src.profile.copy()
        profile.update(dtype=rasterio.uint8, count=1, nodata=None)

    with rasterio.open(output_path / f"{name}.tif", 'w', **profile) as dst:
        dst.write(img, 1)


def calculate_three_band_image(
            input_paths: ThreeBandInputPaths,
            output_path: Path,
            do_zeros=[False, False, False],
            extension: str = "tif"):

        if None in [input_paths.band1, input_paths.band2, input_paths.band3]:
            raise ValueError("Three input paths required for NEN image creation.")

        os.makedirs(output_path, exist_ok=True)

        with rasterio.open(input_paths.band1) as src1, \
            rasterio.open(input_paths.band2) as src2, \
            rasterio.open(input_paths.band3) as src3:

            band1 = src1.read(1)
            band2 = src2.read(1)
            band3 = src3.read(1)

            profile = src1.profile.copy()

        # optional preprocessing
        if do_zeros[0]:
            band1 = np.zeros_like(band1)
        if do_zeros[1]:
            band2 = np.zeros_like(band2)
        if do_zeros[2]:
            band3 = np.zeros_like(band3)


        # stack bands in rasterio format (C, H, W)
        img = np.stack((band1, band2, band3), axis=0).astype(np.uint8)

        QgsMessageLog.logMessage(f"{input_paths.band1}, {input_paths.band2}, {input_paths.band3}")

        output_name = Path(input_paths.band1).stem
        filename = f"{output_name}.{extension}"
        path = output_path / filename

        profile.update(
            dtype=rasterio.uint8,
            count=3,
            nodata=None
        )

        with rasterio.open(path, 'w', **profile) as dst:
            dst.write(img)

def compute_global_stats(tile_dir: Path, band_index: int = 1,
                            fill_threshold: int = 200) -> Tuple[float, float]:
        all_mins = []
        all_maxs = []

        # Debug: confirm the path and what's in it
        tile_paths = sorted(tile_dir.glob("*.tif"))
        if not tile_paths:
            raise FileNotFoundError(
                f"No .tif files found in {tile_dir}\n"
                f"Directory exists: {tile_dir.exists()}\n"
                f"Contents: {list(tile_dir.iterdir())[:10] if tile_dir.exists() else 'N/A'}"
            )

        for tile_path in tqdm.tqdm(tile_paths, desc="Computing global stats"):
            with rasterio.open(tile_path) as src:
                data = src.read(band_index).astype(np.uint16)
                nodata = src.nodata

                if nodata is not None:
                    valid = data[data != int(nodata)]
                else:
                    valid = data.ravel()

                valid = valid[valid >= fill_threshold]

                if valid.size == 0:
                    continue

                all_mins.append(int(np.min(valid)))
                all_maxs.append(int(np.max(valid)))

        if not all_mins:
            raise ValueError(
                f"No valid pixels found in any tile in {tile_dir} "
                f"(band={band_index}, fill_threshold={fill_threshold}). "
                f"Try lowering fill_threshold."
            )

        return (
            float(np.mean(all_mins)),
            float(np.mean(all_maxs))
        )

def cleanup_temp_dir(temp_dir: Path):
    if temp_dir.exists() and temp_dir.is_dir():
        for file in temp_dir.glob("*.tif"):
            file.unlink()
        temp_dir.rmdir()

def tif_to_png(input_path: Path, output_path: Path):
    with rasterio.open(input_path) as src:
        data = src.read()
        profile = src.profile.copy()

    profile.update(driver="PNG", nodata=None)
    for key in ["compress", "tiled", "blockxsize", "blockysize", "interleave"]:
        profile.pop(key, None)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data)


WHITE_PIXEL_VALUE = 65535  # nodata fill used by MicaSense/DJI exports

def is_nodata_tile(tile_path: Path, min_valid_fraction: float = 0.2) -> bool:
    """
    Returns True if the tile contains too little real data to be useful.

    Detection strategy (in priority order):
      1. Band 8 (alpha) present → tile is nodata if fewer than
         `min_valid_fraction` of alpha pixels are non-zero.
         Alpha is binary (0 / 65535) in MicaSense-style exports.
      2. src.nodata is set → count pixels equal to that value in band 1.
      3. White-pixel fallback → nodata pixels are filled with 65535 in all
         bands; count band-1 pixels that are NOT 65535 as valid.

    Parameters
    ----------
    tile_path          : path to the GeoTIFF tile
    min_valid_fraction : fraction of pixels that must be valid to keep the tile
                         (default 0.2 -> remove tiles that are >80% empty)
    """
    with rasterio.open(tile_path) as src:
        total = src.width * src.height
        if total == 0:
            return True

        # Priority 1: alpha channel (most reliable)
        if src.count >= 8:
            alpha = src.read(8)
            valid_fraction = np.sum(alpha > 0) / total
            return valid_fraction < min_valid_fraction

        band1 = src.read(1)

        # Priority 2: rasterio nodata tag
        nodata_val = src.nodata
        if nodata_val is not None:
            valid_fraction = np.sum(band1 != nodata_val) / total
            return valid_fraction < min_valid_fraction

        # Fallback: treat 65535 (white) as nodata fill
        valid_fraction = np.sum(band1 != WHITE_PIXEL_VALUE) / total
        return valid_fraction < min_valid_fraction


def remove_nodata_tiles(tile_dir: Path, min_valid_fraction: float = 0.2) -> int:
    """
    Deletes tiles from `tile_dir` that are predominantly nodata.
    Uses alpha channel (band 8) when present — the correct approach for
    MicaSense/Parrot-style 8-band GeoTIFFs where nodata is not set on
    the raster but the alpha mask is binary (0 / 65535).

    Returns the number of tiles removed.
    """
    tile_paths = sorted(tile_dir.glob("*.tif"))
    removed = 0
    for tile_path in tqdm.tqdm(tile_paths, desc="Removing nodata tiles"):
        if is_nodata_tile(tile_path, min_valid_fraction):
            tile_path.unlink()
            removed += 1
    print(f"Removed {removed}/{len(tile_paths)} nodata tiles from {tile_dir}")
    return removed


def create_nrn_image_from_orthomosaic(orthomosaic_path: Path,
                                    output_path: Path,
                                    global_normalize: bool = False,
                                    tile_size: int = 1024,
                                    overlap: int = 100) -> tuple[Path, Path]:
    """
    Creates a NRN image from an orthomosaic GeoTIFF.

    Parameters:
    orthomosaic_path (str): Path to the input orthomosaic GeoTIFF.
    output_path (str): Path to save the output NRN image.
    indices_to_calculate (list[Indices]): List of indices to calculate and include in the NRN image.

    Returns:
    None
    """
    # Split the orthomosaic

    split_geotiff(orthomosaic_path, output_path / "image_tiles", tile_size=tile_size, overlap=overlap)

    # 2. Remove nodata tiles immediately after splitting
    remove_nodata_tiles(output_path / "image_tiles", min_valid_fraction=0.99)

    if global_normalize:
        # Compute global min/max for each band across all tiles)
        exr_min, exr_max = compute_global_stats(output_path / "image_tiles", band_index=Bands.EXTEND_RED.value)
        exg_min, exg_max = compute_global_stats(output_path / "image_tiles", band_index=Bands.EXTEND_GREEN.value)
        nir_min, nir_max = compute_global_stats(output_path / "image_tiles", band_index=Bands.NIR.value)

    temp_exr_temp_path : Path = output_path / "temp_exr"
    os.makedirs(temp_exr_temp_path, exist_ok=True)

    image_tile_paths : Path = output_path / "image_tiles"

    for img in tqdm.tqdm(sorted(os.listdir(image_tile_paths)), desc="Separating EXTENDED_RED bands from image tiles"):
        if "xml" in img:
            continue
        separate_band(image_tile_paths / img, temp_exr_temp_path, Bands.EXTEND_RED, global_min=exr_min, global_max=exr_max)

    temp_exg_temp_path : Path = output_path / "temp_exg"
    os.makedirs(temp_exg_temp_path, exist_ok=True)

    for img in tqdm.tqdm(sorted(os.listdir(image_tile_paths)), desc="Separating EXTENDED_GREEN bands from image tiles"):
        if "xml" in img:
            continue
        separate_band(image_tile_paths / img, temp_exg_temp_path, Bands.EXTEND_GREEN, global_min=exg_min, global_max=exg_max)

    temp_nir_temp_path : Path = output_path / "temp_nir"
    os.makedirs(temp_nir_temp_path, exist_ok=True)

    for img in tqdm.tqdm(sorted(os.listdir(image_tile_paths)), desc="Separating NIR bands from image tiles"):
        if "xml" in img:
            continue
        separate_band(image_tile_paths / img, temp_nir_temp_path, Bands.NIR, global_min=nir_min, global_max=nir_max)

    temp_ngrdi_path: Path = output_path / "temp_ngrdi"
    os.makedirs(temp_ngrdi_path, exist_ok=True)

    calculate_all_indices(image_tile_paths, temp_ngrdi_path, [Indices.NGRDI_EXTENDED], global_normalize)

    temp_nrn_path: Path = output_path / "temp_nrn"
    os.makedirs(temp_nrn_path, exist_ok=True)

    # create 3-band NRN image
    for img_name in tqdm.tqdm(sorted(os.listdir(image_tile_paths)), desc="NIR_RED_NGRDI image creation"):
        if "xml" in img_name:
            continue

        img_name_nir = temp_nir_temp_path / img_name
        img_name_er = temp_exr_temp_path / img_name
        img_name_ngrdi = temp_ngrdi_path / img_name

        calculate_three_band_image(
            input_paths=ThreeBandInputPaths(
                band1=img_name_nir,   # ← was: input_dir / "nir" / img_name_nir
                band2=img_name_er,    # ← was: input_dir / "extended_red" / img_name_er
                band3=img_name_ngrdi, # ← was: input_dir / "..." / img_name_ngrdi
            ),
            output_path=temp_nrn_path,
            extension="tif"
        )

    temp_nrn_pngs_path: Path = output_path / "temp_nrn_pngs"
    os.makedirs(temp_nrn_pngs_path, exist_ok=True)

    # convert to PNG
    tif_to_png_dir = temp_nrn_path
    for img in tqdm.tqdm(sorted(os.listdir(tif_to_png_dir)), desc="Converting NRN tiles from TIFF to PNG"):
        if "xml" in img:
            continue
        tif_to_png(tif_to_png_dir / img, temp_nrn_pngs_path / img.replace(".tif", ".png"))

    #delete temp band directories
    for temp_dir in tqdm.tqdm([temp_exr_temp_path, temp_exg_temp_path, temp_nir_temp_path, temp_nrn_path, temp_ngrdi_path], desc="Cleaning up temporary directories"):
        cleanup_temp_dir(temp_dir)

    return output_path / "image_tiles", temp_nrn_pngs_path

if __name__ == "__main__":
    orthomosaic_path = Path("/home/fildo/SDU/MasterThesis/Orthomosaics/20250827_Bjørnkjærvej_TestFlight_2_small.tif")
    output_path = Path("/home/fildo/SDU/MasterThesis/Orthomosaics/test_output")

    temp_nrn_pngs = create_nrn_image_from_orthomosaic(orthomosaic_path, output_path, global_normalize=True)
