"""
Composite Orthomosaic Generator
================================
Reads an 8-band orthomosaic (Red, Green, Blue, Extended Green, Extended Red,
Red Edge, NIR, Alpha) and writes a new 3-band uint16 GeoTIFF containing:
    Band 1 – NIR           (passthrough, uint16)
    Band 2 – Extended Red  (passthrough, uint16)
    Band 3 – NGRDI         (float → clamped [-1,1] → stretched [0, 65535] → uint16)

Usage:
    python generate_composite_orthomosaic.py <input.tif> <output.tif>

Dependencies:
    pip install rasterio numpy
"""

import sys
import numpy as np
import rasterio
from rasterio.transform import from_bounds


# ---------------------------------------------------------------------------
# Band indices (1-based, as stored in the file)
# ---------------------------------------------------------------------------
BAND_RED            = 1
BAND_GREEN          = 2
BAND_BLUE           = 3
BAND_EXTENDED_GREEN = 4
BAND_EXTENDED_RED   = 5
BAND_RED_EDGE       = 6
BAND_NIR            = 7
BAND_ALPHA          = 8

UINT16_MAX = 65535


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def read_band_as_float(src: rasterio.DatasetReader, band_index: int) -> np.ndarray:
    """Read a single band and return it as a float64 array."""
    data = src.read(band_index).astype(np.float64)
    return data


def compute_ngrdi(green: np.ndarray, red: np.ndarray) -> np.ndarray:
    """
    NGRDI = (Green - Red) / (Green + Red)

    Where the denominator is zero the index is set to 0.0 (neutral).
    """
    denominator = green + red
    # Avoid division by zero
    valid = denominator != 0
    ngrdi = np.where(valid, (green - red) / denominator, 0.0)
    return ngrdi


def clamp(array: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Clamp all values to [lo, hi]."""
    return np.clip(array, lo, hi)


def histogram_stretch(array: np.ndarray,
                      src_min: float, src_max: float,
                      dst_min: float, dst_max: float) -> np.ndarray:
    """
    Linearly map values in [src_min, src_max] → [dst_min, dst_max].
    """
    if src_max == src_min:
        # Degenerate case – return midpoint of destination range
        return np.full_like(array, (dst_min + dst_max) / 2.0)
    return (array - src_min) / (src_max - src_min) * (dst_max - dst_min) + dst_min


# ---------------------------------------------------------------------------
# Main processing function
# ---------------------------------------------------------------------------

def generate_composite(input_path: str, output_path: str) -> None:
    print(f"[INFO] Opening input:  {input_path}")

    with rasterio.open(input_path) as src:
        n_bands = src.count
        print(f"[INFO] Input has {n_bands} band(s), dtype={src.dtypes[0]}, "
              f"size={src.width}x{src.height}")

        if n_bands < 7:
            raise ValueError(
                f"Expected at least 7 bands (NIR at band 7), found {n_bands}."
            )

        # ── 1. Read required bands ──────────────────────────────────────────
        print("[INFO] Reading bands …")
        nir          = read_band_as_float(src, BAND_NIR)           # band 7
        extended_red = read_band_as_float(src, BAND_EXTENDED_RED)  # band 5
        green        = read_band_as_float(src, BAND_GREEN)          # band 2
        red          = read_band_as_float(src, BAND_RED)            # band 1

        # ── 2. Compute NGRDI ────────────────────────────────────────────────
        print("[INFO] Computing NGRDI …")
        ngrdi = compute_ngrdi(green, red)
        print(f"       Raw NGRDI  – min: {ngrdi.min():.4f}  max: {ngrdi.max():.4f}")

        # ── 3. Clamp to [-1, 1] ─────────────────────────────────────────────
        ngrdi_clamped = clamp(ngrdi, -1.0, 1.0)
        print(f"       After clamp – min: {ngrdi_clamped.min():.4f}  max: {ngrdi_clamped.max():.4f}")

        # ── 4. Stretch [-1, 1] → [0, 65535] ────────────────────────────────
        ngrdi_stretched = histogram_stretch(
            ngrdi_clamped,
            src_min=-1.0, src_max=1.0,
            dst_min=0.0,  dst_max=float(UINT16_MAX)
        )
        print(f"       After stretch – min: {ngrdi_stretched.min():.1f}  max: {ngrdi_stretched.max():.1f}")

        # ── 5. Round and cast everything to uint16 ──────────────────────────
        nir_u16          = np.round(nir).astype(np.uint16)
        extended_red_u16 = np.round(extended_red).astype(np.uint16)
        ngrdi_u16        = np.round(ngrdi_stretched).astype(np.uint16)

        # ── 6. Build output metadata ────────────────────────────────────────
        profile = src.profile.copy()
        profile.update(
            count=3,
            dtype="uint16",
            compress="lzw",          # lossless, space-efficient
            predictor=2,             # horizontal differencing for uint data
            tiled=True,
            blockxsize=256,
            blockysize=256,
            photometric="rgb",       # three-band → stored as RGB container
        )
        # Remove alpha-related settings from the source profile if present
        profile.pop("alpha", None)

        # ── 7. Write output ─────────────────────────────────────────────────
        print(f"[INFO] Writing output: {output_path}")
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(nir_u16,          1)   # Band 1 → NIR
            dst.write(extended_red_u16, 2)   # Band 2 → Extended Red
            dst.write(ngrdi_u16,        3)   # Band 3 → NGRDI (stretched)

            # Tag the bands so downstream tools know what each one holds
            dst.update_tags(1, name="NIR")
            dst.update_tags(2, name="Extended Red")
            dst.update_tags(3, name="NGRDI (stretched 0-65535)")

        print("[INFO] Done ✓")
        print(f"       Output bands:")
        print(f"         1 – NIR           (uint16, 0-{UINT16_MAX})")
        print(f"         2 – Extended Red  (uint16, 0-{UINT16_MAX})")
        print(f"         3 – NGRDI         (uint16, 0-{UINT16_MAX})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_composite_orthomosaic.py <input.tif> <output.tif>")
        sys.exit(1)

    input_path  = sys.argv[1]
    output_path = sys.argv[2]

    generate_composite(input_path, output_path)
