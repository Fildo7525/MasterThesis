"""
gmm_pretrain.py
---------------
Drop-in replacement for ngrvi_pretrain.py that uses a Gaussian Mixture Model
(GMM) for one-class anomaly detection.

Pipeline inside GMMDetector
-----------------------------
    StandardScaler → PCA → GaussianMixture

PCA is fitted on the scaled training data.  The number of components is chosen
by the `pca_variance` parameter (default 0.95 = keep enough components to
explain 95 % of variance).  The fitted PCA object is exposed as `detector.pca_`
so run_diagnostics.py can plot loadings and the PC1/PC2 decision boundary.

How one-class detection works
-------------------------------
The GMM is trained only on inlier data and learns the density of the inlier
distribution in PCA space.  At inference, samples with a log-likelihood below
a threshold (set automatically from the training distribution) are outliers.

Score convention (same as BaseDetector)
----------------------------------------
    score(X)   → log-likelihood  (higher = more normal)
    predict(X) → +1 inlier, -1 outlier

Usage
-----
    python gmm_pretrain.py

Saved .joblib structure:
    {
        "pipeline":           GMMDetector,
        "band_indices":       list[Bands] | None,
        "vegetation_indices": list[Indices] | None,
    }
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from dataclasses import dataclass

import joblib
import numpy as np
import rasterio
import geopandas as gpd
import cv2 as cv
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds, Window, transform
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from base_detector import BaseDetector
from create_indexes import Bands, Indices, compute_index, scale_to_uint16
from features.features import FeatureExtractor

# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------

N_COMPONENTS  = 3      # number of Gaussians — tune via BIC/AIC
COVARIANCE    = "full" # "full" | "tied" | "diag" | "spherical"
MAX_ITER      = 200
CONTAMINATION = 0.01   # fraction of training samples treated as outliers
RANDOM_STATE  = 42

PCA_VARIANCE  = 0.95   # fraction of variance to retain after PCA

UINT16_MAX = 65_535

OUTPUT_PATH = Path.home() / "SDU/MasterThesis/OpenCV/gmm_output_nrn"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

BANDS_TO_USE   = [Bands.EXTEND_RED, Bands.NIR]
INDICES_TO_USE = [Indices.NGRDI]


# ---------------------------------------------------------------------------
# Detector wrapper
# ---------------------------------------------------------------------------

class GMMDetector(BaseDetector):
    """
    One-class detector: StandardScaler → PCA → GaussianMixture.

    Attributes exposed for diagnostics
    ------------------------------------
    scaler_   : fitted StandardScaler
    pca_      : fitted PCA  (components_, explained_variance_ratio_, etc.)
    model     : fitted GaussianMixture

    The decision threshold is set automatically during fit() from the
    contamination-th percentile of training log-likelihoods, mirroring
    how sklearn IsolationForest sets its own offset.

    Tip: call GMMDetector.select_n_components(X) to pick n_components via BIC.
    """

    def __init__(
        self,
        n_components:    int   = N_COMPONENTS,
        covariance_type: str   = COVARIANCE,
        max_iter:        int   = MAX_ITER,
        contamination:   float = CONTAMINATION,
        random_state:    int   = RANDOM_STATE,
        pca_variance:    float = PCA_VARIANCE,
    ):
        self.n_components    = n_components
        self.covariance_type = covariance_type
        self.contamination   = contamination

        self.scaler_ = StandardScaler()
        self.pca_    = PCA(n_components=pca_variance, random_state=random_state)
        self.model   = GaussianMixture(
            n_components    = n_components,
            covariance_type = covariance_type,
            max_iter        = max_iter,
            random_state    = random_state,
            n_init          = 3,
        )
        self._threshold: float | None = None
        self._fitted = False

    # ------------------------------------------------------------------
    # BaseDetector API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "GMMDetector":
        """Fit scaler → PCA → GMM, then auto-compute the decision threshold."""
        X_scaled = self.scaler_.fit_transform(X)
        X_pca    = self.pca_.fit_transform(X_scaled)
        self.model.fit(X_pca)

        train_scores    = self.model.score_samples(X_pca)
        self._threshold = float(np.percentile(train_scores, 100 * self.contamination))
        self._fitted    = True

        n_kept    = self.pca_.n_components_
        total_var = self.pca_.explained_variance_ratio_.sum()
        print(f"   PCA: {X.shape[1]} features → {n_kept} components "
              f"({total_var * 100:.1f} % variance retained)")
        print(f"   GMM threshold (log-likelihood): {self._threshold:.4f}  "
              f"(contamination={self.contamination})")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns +1 (inlier) or -1 (outlier)."""
        self._check_fitted(self._fitted, "GMMDetector")
        return np.where(self.score(X) >= self._threshold, 1, -1).astype(int)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Log-likelihood per sample in PCA space. Higher = more normal."""
        self._check_fitted(self._fitted, "GMMDetector")
        return self.model.score_samples(self._transform(X))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _transform(self, X: np.ndarray) -> np.ndarray:
        """Scale then project through PCA — used by predict() and score()."""
        return self.pca_.transform(self.scaler_.transform(X))

    # ------------------------------------------------------------------
    # Utility: Gaussian component selection via BIC
    # ------------------------------------------------------------------

    @staticmethod
    def select_n_components(
        X: np.ndarray,
        max_components:  int   = 10,
        covariance_type: str   = COVARIANCE,
        pca_variance:    float = PCA_VARIANCE,
        random_state:    int   = RANDOM_STATE,
    ) -> dict[int, float]:
        """
        Fit GMMs with n = 1 … max_components (in PCA space) and return BIC scores.
        Lower BIC = better.  Pick the elbow.

        Example
        -------
            bic  = GMMDetector.select_n_components(X)
            best = min(bic, key=bic.get)
        """
        scaler   = StandardScaler()
        pca      = PCA(n_components=pca_variance, random_state=random_state)
        X_pca    = pca.fit_transform(scaler.fit_transform(X))
        bic: dict[int, float] = {}

        for n in range(1, max_components + 1):
            gmm    = GaussianMixture(n_components=n, covariance_type=covariance_type,
                                     random_state=random_state, n_init=3)
            gmm.fit(X_pca)
            bic[n] = gmm.bic(X_pca)
            print(f"   n_components={n:2d}  BIC={bic[n]:.1f}")

        return bic


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
        n_components:       int              = N_COMPONENTS,
        covariance_type:    str              = COVARIANCE,
        pca_variance:       float            = PCA_VARIANCE,
        band_indices:       list[Bands]   | None = None,
        vegetation_indices: list[Indices] | None = None,
        rectangle:          bool             = False,
    ):
        self.pipeline = GMMDetector(
            n_components    = n_components,
            covariance_type = covariance_type,
            pca_variance    = pca_variance,
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

    def train(
        self,
        ortho_path:        Path,
        shapefile_path:    Path,
        limit:             float = 0.8,
        select_components: bool  = False,
    ):
        """
        Parameters
        ----------
        select_components : bool
            Run BIC selection (n=1…10) before fitting and print a recommendation.
            Useful for first-time tuning.
        """
        print("── Extracting features ───────────────────────────────────")
        X = self.build_feature_matrix(
            ortho_path, shapefile_path,
            self.band_indices, self.vegetation_indices,
            limit=limit,
        )
        self.last_X = X
        print(f"   Feature matrix: {X.shape[0]} × {X.shape[1]}")

        if select_components:
            print("\n── BIC component selection ──────────────────────────────")
            bic  = GMMDetector.select_n_components(X, pca_variance=self.pipeline.pca_.n_components
                                                   if hasattr(self.pipeline.pca_, "n_components_")
                                                   else 0.95)
            best = min(bic, key=bic.get)
            print(f"   → Recommended n_components = {best}  (lowest BIC)")
            self.pipeline = GMMDetector(
                n_components    = best,
                covariance_type = self.pipeline.covariance_type,
            )

        print("\n── Fitting GMM (with PCA) ───────────────────────────────")
        self.pipeline.fit(X)

        preds = self.pipeline.predict(X)
        n_in  = (preds == 1).sum()
        print(f"   Training support: {n_in}/{len(preds)} classified as inlier")

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

def get_feature_names():
    features = FeatureExtractor.get_feature_names()
    feature_names = []
    if BANDS_TO_USE is not None:
        for band in BANDS_TO_USE:
            for feat in features:
                feature_names.append(band.name + "_" + feat)
    if INDICES_TO_USE is not None:
        for band in INDICES_TO_USE:
            for feat in features:
                feature_names.append(band.name + "_" + feat)
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
        n_components       = N_COMPONENTS,
        covariance_type    = COVARIANCE,
        pca_variance       = PCA_VARIANCE,
        band_indices       = BANDS_TO_USE,
        vegetation_indices = INDICES_TO_USE,
        rectangle          = False,
    )

    for cfg in configs:
        trainer.train(
            ortho_path        = cfg.ortho_path,
            shapefile_path    = cfg.shapefile_path,
            limit             = 0.8,
            select_components = False,  # set True on first run to pick best n
        )

    trainer.dump(OUTPUT_PATH / "gmm_model.joblib")

    feature_names = get_feature_names()

    from run_diagnostics import plot_feature_matrix, plot_pca_importance, plot_pca_scatter
    plot_feature_matrix(trainer, OUTPUT_PATH / "feature_matrix.png",
                        feature_names=feature_names, max_features=10)
    plot_pca_importance(trainer, OUTPUT_PATH / "pca_importance.png",
                        feature_names=feature_names)
    plot_pca_scatter(trainer, OUTPUT_PATH / "pca_scatter.png")
