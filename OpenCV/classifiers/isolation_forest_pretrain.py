"""
isolation_forest_pretrain.py
-----------------------------
Drop-in replacement for ngrvi_pretrain.py that uses an Isolation Forest
instead of a One-Class SVM.

Usage
-----
    python isolation_forest_pretrain.py

The saved .joblib file has the same structure as the original:
    {
        "pipeline":          IsolationForestDetector,
        "band_indices":      list[Bands] | None,
        "vegetation_indices": list[Indices] | None,
    }

NgrviApproach loads the file and calls:
    pipeline.predict(X)           → +1 / -1
    pipeline.decision_function(X) → anomaly score (higher = more normal)

Both are implemented on IsolationForestDetector via the BaseDetector ABC.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import rasterio
import geopandas as gpd
import cv2 as cv
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds, Window, transform
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from base_detector import BaseDetector
from create_indexes import Bands, Indices, compute_index, scale_to_uint16
from features.features import FeatureExtractor

# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------

N_ESTIMATORS           = 200    # more trees → more stable, slower training
MAX_SAMPLES            = "auto" # "auto" = min(256, n_samples)
CONTAMINATION          = 0.01   # expected fraction of outliers in training data
                                # keep small since your training set is pure inliers
RANDOM_STATE           = 42

UINT16_MAX = 65_535

OUTPUT_PATH = Path.home() / "SDU/MasterThesis/OpenCV/iforest_output_rgb"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

BANDS_TO_USE   = [Bands.RED, Bands.GREEN, Bands.BLUE]
INDICES_TO_USE = None


# ---------------------------------------------------------------------------
# Detector wrapper
# ---------------------------------------------------------------------------

class IsolationForestDetector(BaseDetector):
    """
    Wraps sklearn IsolationForest + StandardScaler behind the BaseDetector
    interface so NgrviApproach.process_window() needs no changes.

    Score convention
    ----------------
    sklearn IsolationForest.score_samples() returns the negative average
    path-length anomaly score: MORE NEGATIVE = more anomalous.
    We negate it so that HIGHER = more normal, matching the OCSVM convention
    used in the existing debug prints.
    """

    def __init__(
        self,
        n_estimators: int   = N_ESTIMATORS,
        max_samples         = MAX_SAMPLES,
        contamination       = CONTAMINATION,
        random_state: int   = RANDOM_STATE,
    ):
        self.scaler = StandardScaler()
        self.model  = IsolationForest(
            n_estimators  = n_estimators,
            max_samples   = max_samples,
            contamination = contamination,
            random_state  = random_state,
            n_jobs        = -1,
        )
        self._fitted = False

    # ------------------------------------------------------------------
    # BaseDetector API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns +1 (inlier) or -1 (outlier)."""
        self._check_fitted(self._fitted, "IsolationForestDetector")
        return self.model.predict(self.scaler.transform(X))

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Higher score → more normal.
        sklearn's score_samples returns values in roughly [-0.5, 0.5];
        negating puts it in [−0.5, +0.5] with +0.5 = most normal.
        """
        self._check_fitted(self._fitted, "IsolationForestDetector")
        # score_samples: higher = more normal in sklearn convention already
        return self.model.score_samples(self.scaler.transform(X))


# ---------------------------------------------------------------------------
# Pretrainer  (mirrors ngrvi_pretrain.Pretrainer exactly)
# ---------------------------------------------------------------------------

@dataclass
class PretrainConfig:
    ortho_path:     Path
    shapefile_path: Path


class Pretrainer:
    def __init__(
        self,
        n_estimators: int = N_ESTIMATORS,
        band_indices:        list[Bands]   | None = None,
        vegetation_indices:  list[Indices] | None = None,
        rectangle: bool = False,
    ):
        self.detector          = IsolationForestDetector(n_estimators=n_estimators)
        self.band_indices      = band_indices
        self.vegetation_indices = vegetation_indices
        self.rectangle         = rectangle
        self.last_X: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Geometry helpers  (identical to original Pretrainer)
    # ------------------------------------------------------------------

    def polygon_to_pixel_mask(self, geometry, src: rasterio.DatasetReader):
        minx, miny, maxx, maxy = geometry.bounds
        window = from_bounds(minx, miny, maxx, maxy, src.transform)
        window = window.round_lengths().round_offsets()
        window = window.intersection(Window(0, 0, src.width, src.height))

        win_transform = transform(window, src.transform)
        win_height    = int(window.height)
        win_width     = int(window.width)

        outside    = geometry_mask(
            [geometry], transform=win_transform, invert=False,
            out_shape=(win_height, win_width),
        )
        return window, ~outside

    def create_ngrvi_mask(self, bands: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        list_bands      = [bands[i] for i in range(bands.shape[0])]
        index           = compute_index(Indices.NGRVI.name, list_bands)
        ngrdi_u16       = scale_to_uint16(index, Indices.NGRVI.name)
        threshold_value = UINT16_MAX * 0.016
        mask            = np.zeros_like(ngrdi_u16)
        cv.threshold(ngrdi_u16, threshold_value, UINT16_MAX, cv.THRESH_BINARY, dst=mask)
        return mask, ngrdi_u16

    def extract_vector_from_polygon(
        self,
        geometry,
        src:               rasterio.DatasetReader,
        extractor:         FeatureExtractor,
        band_indices:      list[Bands]   | None,
        vegetation_indices: list[Indices] | None,
    ) -> np.ndarray | None:

        window, pixel_mask = self.polygon_to_pixel_mask(geometry, src)
        if pixel_mask.sum() < 9:
            return None

        scale = 65535
        band_indices_1based = list(range(1, 8))
        actual_indices      = list(range(src.count))

        chip  = src.read(band_indices_1based, window=window).astype(np.float32) / scale
        bands = [chip[i] for i in range(len(actual_indices))]

        ngrvi_mask, _ = self.create_ngrvi_mask(chip)

        results = extractor.process_multiband(
            bands,
            band_indices       = band_indices,
            mask               = ngrvi_mask,
            rectangle          = self.rectangle,
            vegetation_indices = vegetation_indices,
        )

        values = []
        for _, feats in sorted(results.items()):
            if feats is not None:
                for _, val in sorted(feats.items()):
                    values.append(float(val))

        return np.array(values) if values else None

    # ------------------------------------------------------------------
    # Build feature matrix
    # ------------------------------------------------------------------

    def build_feature_matrix(
        self,
        ortho_path:         Path,
        shapefile_path:     Path,
        band_indices:       list[Bands]   | None,
        vegetation_indices: list[Indices] | None,
        limit:              float = 0.8,
    ) -> np.ndarray:

        extractor  = FeatureExtractor()
        gdf        = gpd.read_file(shapefile_path)
        limit      = float(np.clip(limit, 0, 1))
        rows: list = []
        skipped    = 0

        picked_gdf = gdf.sample(frac=limit, random_state=42)
        pth        = OUTPUT_PATH / f"picked_polygons_{ortho_path.stem}_limit_{limit}.shp"
        picked_gdf.to_file(pth)
        print(f"  Saved picked polygons → {pth}")
        print(f"  Picked: {len(picked_gdf)} / {len(gdf)}  (limit={limit})")

        with rasterio.open(ortho_path) as src:
            if picked_gdf.crs != src.crs:
                print(f"  Reprojecting {picked_gdf.crs} → {src.crs}")
                picked_gdf = picked_gdf.to_crs(src.crs)

            for idx, row in tqdm(picked_gdf.iterrows(), total=len(picked_gdf), desc="Polygons"):
                geom = row.geometry
                if geom is None or geom.is_empty:
                    skipped += 1
                    continue

                vec = self.extract_vector_from_polygon(
                    geom, src, extractor, band_indices, vegetation_indices
                )
                if vec is None:
                    skipped += 1
                    continue
                rows.append(vec)

        if skipped:
            print(f"  Skipped {skipped} polygon(s)")
        if not rows:
            raise RuntimeError("No valid feature vectors extracted.")

        return np.vstack(rows)

    # ------------------------------------------------------------------
    # Train & save
    # ------------------------------------------------------------------

    def train(self, ortho_path: Path, shapefile_path: Path, limit: float = 0.8):
        print("── Extracting features ───────────────────────────────────")
        X = self.build_feature_matrix(
            ortho_path, shapefile_path,
            self.band_indices, self.vegetation_indices,
            limit=limit,
        )
        self.last_X = X
        print(f"   Feature matrix: {X.shape[0]} × {X.shape[1]}")

        print("\n── Fitting Isolation Forest ─────────────────────────────")
        self.detector.fit(X)

        preds = self.detector.predict(X)
        n_in  = (preds == 1).sum()
        print(f"   Training support: {n_in}/{len(preds)} classified as inlier")

    def dump(self, out_path: Path):
        meta = {
            "pipeline":           self.detector,   # NgrviApproach key
            "band_indices":       self.band_indices,
            "vegetation_indices": self.vegetation_indices,
        }
        joblib.dump(meta, out_path)
        print(f"\n── Model saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI
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

    trainer = Pretrainer(
        n_estimators       = N_ESTIMATORS,
        band_indices       = BANDS_TO_USE,
        vegetation_indices = INDICES_TO_USE,
        rectangle          = False,
    )

    for cfg in configs:
        trainer.train(ortho_path=cfg.ortho_path, shapefile_path=cfg.shapefile_path, limit=0.8)

    trainer.dump(OUTPUT_PATH / "iforest_model.joblib")
