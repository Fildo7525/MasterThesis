"""
shapefile_contour_adjuster.py
------------------------------
Copies a ground-truth shapefile and replaces every polygon's geometry with
the boundary of the NGRVI vegetation mask generated from the corresponding
orthomosaic chip.

Pipeline per polygon
--------------------
    1. Extract the raster chip that covers the polygon's bounding box.
    2. Build the NGRVI binary mask (same threshold as svm_pretrain.py).
    3. Find the *largest* OpenCV contour in the mask.
    4. Convert pixel-space contour vertices → geo-coordinates via the
       window's Affine transform.
    5. Store the resulting Shapely polygon in the adjusted shapefile.

Polygons for which no valid contour is found are kept unchanged (or
optionally dropped — see SKIP_NO_CONTOUR below).

Output
------
For each PretrainConfig an adjusted shapefile is written next to the
original, with the suffix `_contour_adjusted` added to the stem, e.g.:
    small_obb_test.shp  →  small_obb_test_contour_adjusted.shp

Tuneable constants at the top of the file mirror svm_pretrain.py so
changing the threshold in one place stays in sync with the other.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from dataclasses import dataclass

import cv2 as cv
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds, Window, transform as win_transform_fn
from shapely.geometry import Polygon, MultiPolygon
from tqdm import tqdm

from create_indexes import Bands, Indices, compute_index, scale_to_uint16

# ---------------------------------------------------------------------------
# Tuneable constants  (keep in sync with svm_pretrain.py)
# ---------------------------------------------------------------------------

UINT16_MAX = 65_535

# Vegetation index used for masking
VI = Indices.NGRVI

# Per-index thresholds (uint16 space)
THRESHOLDS = {
    Indices.NGRVI: UINT16_MAX * 0.016,
    Indices.NDVI:  UINT16_MAX * 0.820,
    Indices.EXGR:  UINT16_MAX * 0.5009,
    Indices.NGRDI: UINT16_MAX * 0.534065766,
    Indices.MGRVI: UINT16_MAX * 0.5,
}

# Minimum contour area in pixels to be considered valid
MIN_CONTOUR_AREA_PX = 4

# If True, polygons with no valid contour are omitted from the output
# shapefile.  If False they are written with their original geometry.
SKIP_NO_CONTOUR = False

# Output directory
OUTPUT_PATH = Path.home() / "SDU/MasterThesis/OpenCV/contour_adjusted_shapefiles"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Config  (mirrors PretrainConfig in svm_pretrain.py)
# ---------------------------------------------------------------------------

@dataclass
class PretrainConfig:
    ortho_path:     Path
    shapefile_path: Path


# ---------------------------------------------------------------------------
# Core adjuster
# ---------------------------------------------------------------------------

class ShapefileContourAdjuster:
    """
    For each polygon in the source shapefile, replaces its geometry with
    the boundary of the NGRVI mask contour extracted from the orthomosaic.
    """

    def __init__(
        self,
        vi:             Indices = VI,
        thresholds:     dict    = THRESHOLDS,
        min_area_px:    int     = MIN_CONTOUR_AREA_PX,
        skip_no_contour: bool   = SKIP_NO_CONTOUR,
    ):
        self.vi              = vi
        self.thresholds      = thresholds
        self.min_area_px     = min_area_px
        self.skip_no_contour = skip_no_contour

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _polygon_window(
        self,
        geometry,
        src: rasterio.DatasetReader,
    ) -> tuple[Window, rasterio.Affine]:
        """Return (window, window_transform) for the polygon's bounding box."""
        minx, miny, maxx, maxy = geometry.bounds
        window = from_bounds(minx, miny, maxx, maxy, src.transform)
        window = window.round_lengths().round_offsets()
        window = window.intersection(Window(0, 0, src.width, src.height))
        affine = win_transform_fn(window, src.transform)
        return window, affine

    def _ngrvi_mask(self, chip: np.ndarray) -> np.ndarray:
        """
        Compute the VI mask for a chip of shape (bands, H, W).
        Returns a uint16 binary mask (0 / UINT16_MAX).
        """
        list_bands      = [chip[i] for i in range(chip.shape[0])]
        index           = compute_index(self.vi, list_bands)
        index_u16       = scale_to_uint16(index, self.vi)
        threshold_value = self.thresholds[self.vi]
        mask            = np.zeros_like(index_u16)
        cv.threshold(index_u16, threshold_value, UINT16_MAX, cv.THRESH_BINARY, dst=mask)
        return mask

    def _largest_contour(
        self, mask: np.ndarray
    ) -> np.ndarray | None:
        """
        Return the largest contour found in the uint16 mask, or None.
        The returned array has shape (N, 1, 2) with integer pixel coords.
        """
        # findContours needs uint8
        mask_u8 = (mask > 0).astype(np.uint8) * 255
        contours, _ = cv.findContours(
            mask_u8, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        # Pick the contour with the largest area
        best = max(contours, key=cv.contourArea)
        if cv.contourArea(best) < self.min_area_px:
            return None
        return best

    def _contour_to_geo_polygon(
        self,
        contour:        np.ndarray,
        window_affine:  rasterio.Affine,
    ) -> Polygon | None:
        """
        Return the axis-aligned bounding box of an OpenCV contour as a
        Shapely Polygon in the CRS of the raster.

        cv2.boundingRect gives (x, y, w, h) in pixel space where (x, y)
        is the top-left corner.  Each corner is mapped to geo-coordinates
        via the window's Affine transform:  geo = window_affine * (col, row)
        """
        x, y, w, h = cv.boundingRect(contour)
        if w == 0 or h == 0:
            return None

        # Four corners in pixel space (col, row)
        corners_px = [
            (x,     y    ),   # top-left
            (x + w, y    ),   # top-right
            (x + w, y + h),   # bottom-right
            (x,     y + h),   # bottom-left
        ]
        geo_pts = [window_affine * (float(c), float(r)) for c, r in corners_px]

        try:
            poly = Polygon(geo_pts)
        except Exception:
            return None

        if poly.is_empty or poly.area == 0:
            return None

        return poly

    # Main entry point
    # ------------------------------------------------------------------

    def adjust(
        self,
        ortho_path:     Path,
        shapefile_path: Path,
        output_path:    Path | None = None,
    ) -> Path:
        """
        Read *shapefile_path*, adjust every polygon to its NGRVI contour
        using *ortho_path*, and write the result.

        Parameters
        ----------
        ortho_path     : Path to the orthomosaic (.tif).
        shapefile_path : Path to the ground-truth shapefile.
        output_path    : Destination shapefile path.  Defaults to
                         <shapefile_stem>_contour_adjusted.shp in OUTPUT_PATH.

        Returns
        -------
        Path of the written shapefile.
        """
        if output_path is None:
            output_path = (
                OUTPUT_PATH
                / shapefile_path.parent.name          # e.g. "small"
                / f"{shapefile_path.stem}_contour_adjusted.shp"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        gdf = gpd.read_file(shapefile_path)
        print(f"\n── Processing shapefile: {shapefile_path}")
        print(f"   Polygons in source:   {len(gdf)}")

        adjusted_geoms = []
        kept_indices   = []
        stats          = {"adjusted": 0, "unchanged": 0, "skipped": 0}

        with rasterio.open(ortho_path) as src:
            # Reproject shapefile to raster CRS if needed.
            # Use .equals() rather than != to avoid false mismatches when
            # both sides represent the same EPSG but via different pyproj
            # object representations (e.g. EPSG:25832 != EPSG:25832).
            working_gdf = gdf
            src_crs     = src.crs
            needs_reproject = not gdf.crs.equals(src_crs) if gdf.crs else True
            if needs_reproject:
                print(f"   Reprojecting shapefile: {gdf.crs} → {src_crs}")
                working_gdf = gdf.to_crs(src_crs)

            band_indices_1based = list(range(1, src.count + 1))
            scale               = 65_535.0

            for idx, row in tqdm(
                working_gdf.iterrows(),
                total=len(working_gdf),
                desc="Adjusting polygons",
            ):
                geom = row.geometry
                if geom is None or geom.is_empty:
                    stats["skipped"] += 1
                    continue

                # --- 1. Window & chip ----------------------------------------
                try:
                    window, affine = self._polygon_window(geom, src)
                    if window.height < 1 or window.width < 1:
                        raise ValueError("Zero-size window")
                except Exception as e:
                    print(f"   [warn] polygon {idx}: window error ({e}), keeping original")
                    if not self.skip_no_contour:
                        adjusted_geoms.append(geom)
                        kept_indices.append(idx)
                        stats["unchanged"] += 1
                    else:
                        stats["skipped"] += 1
                    continue

                chip = (
                    src.read(band_indices_1based, window=window).astype(np.float32)
                    / scale
                )

                # --- 2. NGRVI mask -------------------------------------------
                mask = self._ngrvi_mask(chip)

                # --- 3. Largest contour in pixel space -----------------------
                contour = self._largest_contour(mask)

                if contour is None:
                    if not self.skip_no_contour:
                        adjusted_geoms.append(geom)
                        kept_indices.append(idx)
                        stats["unchanged"] += 1
                    else:
                        stats["skipped"] += 1
                    continue

                # --- 4. Convert contour → geo polygon -----------------------
                new_geom = self._contour_to_geo_polygon(contour, affine)

                if new_geom is None:
                    if not self.skip_no_contour:
                        adjusted_geoms.append(geom)
                        kept_indices.append(idx)
                        stats["unchanged"] += 1
                    else:
                        stats["skipped"] += 1
                    continue

                adjusted_geoms.append(new_geom)
                kept_indices.append(idx)
                stats["adjusted"] += 1

        # ------------------------------------------------------------------
        # Write output shapefile (preserve all non-geometry columns)
        # ------------------------------------------------------------------
        out_gdf          = gdf.loc[kept_indices].copy()
        out_gdf.geometry = adjusted_geoms

        # The adjusted geometries live in working_gdf's CRS (which equals
        # src.crs).  If we reprojected, out_gdf still carries the original
        # CRS from gdf, so we must override it.  allow_override=True is safe
        # here because we are *declaring* the CRS of already-transformed
        # coordinates, not re-projecting anything.
        if needs_reproject:
            out_gdf = out_gdf.set_crs(working_gdf.crs, allow_override=True)

        out_gdf.to_file(output_path)

        print(f"\n   Adjusted  : {stats['adjusted']}")
        print(f"   Unchanged : {stats['unchanged']}")
        print(f"   Skipped   : {stats['skipped']}")
        print(f"   ✓ Written → {output_path.absolute()}")
        return output_path


# ---------------------------------------------------------------------------
# CLI / main  (mirrors the __main__ block in svm_pretrain.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    home = Path.home()

    configs = [
        PretrainConfig(
            ortho_path     = home / "SDU/MasterThesis/Orthomosaics/20250827_Bjørnkjærvej_TestFlight_2_small.tif",
            shapefile_path = home / "SDU/MasterThesis/Orthomosaics/shapefiles/small/small_obb_test.shp",
        ),
        PretrainConfig(
            ortho_path     = home / "SDU/MasterThesis/Orthomosaics/20250827_Bjørnkjærvej_TestFlight_2_mid.tif",
            shapefile_path = home / "SDU/MasterThesis/Orthomosaics/shapefiles/mid/mid_obb_test.shp",
        ),
        PretrainConfig(
            ortho_path     = home / "SDU/MasterThesis/Orthomosaics/20250827_Bjørnkjærvej_TestFlight_2_bigger_v2.tif",
            shapefile_path = home / "SDU/MasterThesis/Orthomosaics/shapefiles/large/large_obb_test.shp",
        ),
    ]

    adjuster = ShapefileContourAdjuster(
        vi              = VI,
        thresholds      = THRESHOLDS,
        min_area_px     = MIN_CONTOUR_AREA_PX,
        skip_no_contour = SKIP_NO_CONTOUR,
    )

    for cfg in configs:
        adjuster.adjust(
            ortho_path     = cfg.ortho_path,
            shapefile_path = cfg.shapefile_path,
            # output_path  = ...   ← omit to use the default naming scheme
        )

    print("\n── All done.")
