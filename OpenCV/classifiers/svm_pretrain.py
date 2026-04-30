"""
svm_pretrain.py
---------------
One-Class SVM pretrainer.

Pipeline inside SVMDetector
-----------------------------
    StandardScaler → PCA → OneClassSVM

Follows the same BaseDetector interface as IsolationForestDetector and
GMMDetector, so the saved .joblib file is a drop-in for NgrviApproach:
    {
        "pipeline":           SVMDetector,
        "band_indices":       list[Bands] | None,
        "vegetation_indices": list[Indices] | None,
    }

NgrviApproach calls:
    pipeline.predict(X)           → +1 / -1
    pipeline.decision_function(X) → anomaly score (higher = more normal)

last_X accumulation
--------------------
train() is called once per orthomosaic config.  Each call appends to
self.last_X via np.vstack so that pipeline.fit() at the end of __main__
trains on the combined feature matrix from all configs.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from dataclasses import dataclass

import cv2 as cv
import joblib
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds, Window, transform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from tqdm import tqdm

from base_detector import BaseDetector
from create_indexes import Bands, Indices, compute_index, scale_to_uint16
from features.features import FeatureExtractor

# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------

NU           = 0.001
KERNEL       = "rbf"
PCA_VARIANCE = 0.95    # fraction of variance to retain after PCA

UINT16_MAX = 65_535

OUTPUT_PATH = Path.home() / "SDU/MasterThesis/OpenCV/svm_output_nrn_rgb"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

BANDS_TO_USE   = [Bands.EXTEND_RED, Bands.NIR]# [Bands.EXTEND_RED, Bands.NIR]
INDICES_TO_USE = [Indices.NGRDI]


# ---------------------------------------------------------------------------
# Detector wrapper
# ---------------------------------------------------------------------------

class SVMDetector(BaseDetector):
    """
    Wraps StandardScaler → PCA → OneClassSVM behind the BaseDetector interface
    so NgrviApproach.process_window() needs no changes.

    Attributes exposed for diagnostics (mirrors IsolationForestDetector /
    GMMDetector so run_diagnostics.py works without modification):
        scaler_  : fitted StandardScaler
        pca_     : fitted PCA
        model    : fitted OneClassSVM

    Score convention
    ----------------
    OneClassSVM.decision_function returns signed distances from the hyperplane:
        positive  →  inlier side  (more normal)
        negative  →  outlier side
    This already matches the "higher = more normal" convention used across
    all detectors, so no negation is needed.
    """

    def __init__(
        self,
        nu:           float = NU,
        kernel:       str   = KERNEL,
        pca_variance: float = PCA_VARIANCE,
    ):
        self.scaler_ = StandardScaler()
        self.pca_    = PCA(n_components=pca_variance)
        self.model   = OneClassSVM(kernel=kernel, nu=nu, gamma="scale")
        self._fitted = False

    # ------------------------------------------------------------------
    # BaseDetector API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "SVMDetector":
        X_scaled = self.scaler_.fit_transform(X)
        X_pca    = self.pca_.fit_transform(X_scaled)
        self.model.fit(X_pca)
        self._fitted = True

        n_kept    = self.pca_.n_components_
        total_var = self.pca_.explained_variance_ratio_.sum()
        print(f"   PCA: {X.shape[1]} features → {n_kept} components "
              f"({total_var * 100:.1f} % variance retained)")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns +1 (inlier) or -1 (outlier)."""
        self._check_fitted(self._fitted, "SVMDetector")
        return self.model.predict(self._transform(X))

    def score(self, X: np.ndarray) -> np.ndarray:
        """Signed distance from the hyperplane. Higher = more normal."""
        self._check_fitted(self._fitted, "SVMDetector")
        return self.model.decision_function(self._transform(X))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _transform(self, X: np.ndarray) -> np.ndarray:
        """Scale then project through PCA."""
        return self.pca_.transform(self.scaler_.transform(X))


# ---------------------------------------------------------------------------
# Pretrainer
# ---------------------------------------------------------------------------

@dataclass
class PretrainConfig:
    ortho_path:     Path
    shapefile_path: Path


class Pretrainer:
    def __init__(
        self,
        nu:                 float            = NU,
        kernel:             str              = KERNEL,
        pca_variance:       float            = PCA_VARIANCE,
        band_indices:       list[Bands]   | None = None,
        vegetation_indices: list[Indices] | None = None,
        rectangle:          bool             = False,
    ):
        self.pipeline = SVMDetector(
            nu           = nu,
            kernel       = kernel,
            pca_variance = pca_variance,
        )
        self.band_indices       = band_indices
        self.vegetation_indices = vegetation_indices
        self.rectangle          = rectangle
        self.last_X: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def polygon_to_pixel_mask(self, geometry, src: rasterio.DatasetReader):
        minx, miny, maxx, maxy = geometry.bounds
        window = from_bounds(minx, miny, maxx, maxy, src.transform)
        window = window.round_lengths().round_offsets()
        window = window.intersection(Window(0, 0, src.width, src.height))

        win_transform = transform(window, src.transform)
        win_height    = int(window.height)
        win_width     = int(window.width)

        outside = geometry_mask(
            [geometry],
            transform  = win_transform,
            invert     = False,
            out_shape  = (win_height, win_width),
        )
        return window, ~outside

    def create_ngrvi_mask(self, bands: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        list_bands      = [bands[i, :, :] for i in range(bands.shape[0])]
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

        scale               = 65535
        band_indices_1based = list(range(1, 8))
        actual_indices      = list(range(src.count))

        chip  = src.read(band_indices_1based, window=window).astype(np.float32) / scale
        bands = [band for band in chip[:len(actual_indices)]]

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
        extractor = FeatureExtractor()
        gdf       = gpd.read_file(shapefile_path)
        limit     = float(np.clip(limit, 0, 1))
        rows: list = []
        skipped    = 0

        picked_gdf = gdf.sample(frac=limit, random_state=42)
        pth        = OUTPUT_PATH / f"picked_polygons_{ortho_path.stem}_limit_{limit}.shp"
        picked_gdf.to_file(pth)
        print(f"  Saved picked polygons → {pth.absolute()}")
        print(f"  Picked polygons: {len(picked_gdf)} / {len(gdf)}  (limit={limit})")

        with rasterio.open(ortho_path) as src:
            if picked_gdf.crs != src.crs:
                print(f"  Reprojecting shapefile from {picked_gdf.crs} → {src.crs}")
                picked_gdf = picked_gdf.to_crs(src.crs)

            for idx, row in tqdm(picked_gdf.iterrows(), total=len(picked_gdf), desc="Polygons"):
                geom = row.geometry
                if geom is None or geom.is_empty:
                    print(f"  Skipping polygon {idx} (empty geometry)")
                    skipped += 1
                    continue

                vec = self.extract_vector_from_polygon(
                    geom, src, extractor, band_indices, vegetation_indices
                )
                if vec is None:
                    print(f"  Skipping polygon {idx} (too small or empty after masking)")
                    skipped += 1
                    continue
                rows.append(vec)

        if skipped:
            print(f"  Skipped {skipped} polygon(s) (empty or too small)")
        if not rows:
            raise RuntimeError(
                "No valid feature vectors extracted. "
                "Check that the shapefile overlaps the orthomosaic."
            )

        return np.vstack(rows)

    # ------------------------------------------------------------------
    # Train (accumulate only) & fit & save
    # ------------------------------------------------------------------

    def train(self, ortho_path: Path, shapefile_path: Path, limit: float = 0.8):
        """
        Extract features from one orthomosaic and accumulate into last_X.
        Does NOT fit the model — call fit() after all train() calls so the
        SVM sees the combined feature matrix from every config.
        """
        print("── Extracting features from polygons ────────────────────")
        print("  Processing shapefile:", shapefile_path)

        X = self.build_feature_matrix(
            ortho_path, shapefile_path,
            self.band_indices, self.vegetation_indices,
            limit=limit,
        )

        # Accumulate across multiple train() calls
        if self.last_X is None:
            self.last_X = X
        else:
            self.last_X = np.vstack([self.last_X, X])

        print(f"   Batch: {X.shape[0]} samples × {X.shape[1]} features  "
              f"| Accumulated: {self.last_X.shape[0]} total")

    def fit(self):
        """
        Fit the SVMDetector on the full accumulated feature matrix.
        Call this once after all train() calls.
        """
        if self.last_X is None:
            raise RuntimeError("No data accumulated. Call train() at least once first.")

        print("\n── Fitting One-Class SVM (with PCA) ─────────────────────")
        print(f"   Total training samples: {self.last_X.shape[0]} × {self.last_X.shape[1]}")
        self.pipeline.fit(self.last_X)

        preds = self.pipeline.predict(self.last_X)
        print(f"   Training support: {(preds == 1).sum()}/{len(preds)} in-group")

    def dump(self, out_path: Path):
        meta = {
            "pipeline":           self.pipeline,
            "band_indices":       self.band_indices,
            "vegetation_indices": self.vegetation_indices,
        }
        joblib.dump(meta, out_path)
        print(f"\n── Model saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_feature_names() -> list[str]:
    features      = FeatureExtractor.get_feature_names()
    feature_names = []
    if BANDS_TO_USE is not None:
        for band in BANDS_TO_USE:
            for feat in features:
                feature_names.append(band.name + "_" + feat)
    if INDICES_TO_USE is not None:
        for idx in INDICES_TO_USE:
            for feat in features:
                feature_names.append(idx.name + "_" + feat)
    return feature_names


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
        nu                 = NU,
        kernel             = KERNEL,
        pca_variance       = PCA_VARIANCE,
        band_indices       = BANDS_TO_USE,
        vegetation_indices = INDICES_TO_USE,
        rectangle          = False,
    )

    # Phase 1: accumulate feature vectors from all orthomosaics
    for cfg in configs:
        trainer.train(
            ortho_path     = cfg.ortho_path,
            shapefile_path = cfg.shapefile_path,
            limit          = 0.8,
        )

    # Phase 2: fit once on the full combined matrix
    trainer.fit()

    # Phase 3: save
    trainer.dump(OUTPUT_PATH / "pretrain_output_model.joblib")

    # Phase 4: diagnostics
    feature_names = get_feature_names()

    from run_diagnostics import plot_feature_matrix, plot_pca_importance, plot_pca_scatter
    plot_feature_matrix(trainer, OUTPUT_PATH / "feature_matrix.png",
                        feature_names=feature_names, max_features=5)
    plot_pca_importance(trainer, OUTPUT_PATH / "pca_importance.png",
                        feature_names=feature_names, pca_variance=PCA_VARIANCE)
    plot_pca_scatter(trainer, OUTPUT_PATH / "pca_scatter.png")
