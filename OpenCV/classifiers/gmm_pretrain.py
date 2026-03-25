"""
gmm_pretrain.py
---------------
Drop-in replacement for ngrvi_pretrain.py that uses a Gaussian Mixture Model
(GMM) for one-class anomaly detection.

How GMM works for one-class detection
--------------------------------------
The GMM is trained only on inlier data and learns the density of the inlier
distribution.  At inference, we compute the log-likelihood of each sample
under the learned model.  Samples with a log-likelihood below a threshold are
classified as outliers.

The threshold is computed automatically after fitting: we score all training
samples, sort them, and take the `contamination`-th percentile as the cut-off.
This mirrors how sklearn's IsolationForest sets its own threshold.

Score convention (same as BaseDetector)
----------------------------------------
    score(X)   → higher = more normal  (log-likelihood, less negative = more normal)
    predict(X) → +1 inlier, -1 outlier

Usage
-----
    python gmm_pretrain.py

Saved .joblib structure (same as ngrvi_pretrain.py):
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
from pathlib import Path

import joblib
import numpy as np
import rasterio
import geopandas as gpd
import cv2 as cv
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds, Window, transform
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from base_detector import BaseDetector
from create_indexes import Bands, Indices, compute_index, scale_to_uint16
from features.features import FeatureExtractor

# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------

N_COMPONENTS  = 3      # number of Gaussians — tune via BIC/AIC (see fit() note)
COVARIANCE    = "full" # "full" | "tied" | "diag" | "spherical"
MAX_ITER      = 200
CONTAMINATION = 0.01   # fraction of training samples treated as outliers
                        # when auto-computing the decision threshold
RANDOM_STATE  = 42

UINT16_MAX = 65_535

OUTPUT_PATH = Path.home() / "SDU/MasterThesis/OpenCV/gmm_output_rgb"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

BANDS_TO_USE   = [Bands.RED, Bands.GREEN, Bands.BLUE]
INDICES_TO_USE = None


# ---------------------------------------------------------------------------
# Detector wrapper
# ---------------------------------------------------------------------------

class GMMDetector(BaseDetector):
    """
    One-class detector backed by a Gaussian Mixture Model.

    The decision threshold is set automatically during fit() based on the
    log-likelihood distribution of the training data and the chosen
    contamination level.

    Tip: if you are unsure how many Gaussian components to use, call
    GMMDetector.select_n_components(X) which returns a dict of BIC scores
    for n = 1 … max_components so you can pick the elbow.
    """

    def __init__(
        self,
        n_components:  int   = N_COMPONENTS,
        covariance_type: str = COVARIANCE,
        max_iter:      int   = MAX_ITER,
        contamination: float = CONTAMINATION,
        random_state:  int   = RANDOM_STATE,
    ):
        self.n_components    = n_components
        self.covariance_type = covariance_type
        self.contamination   = contamination

        self.scaler    = StandardScaler()
        self.model     = GaussianMixture(
            n_components    = n_components,
            covariance_type = covariance_type,
            max_iter        = max_iter,
            random_state    = random_state,
            n_init          = 3,   # multiple restarts for stability
        )
        self._threshold: float | None = None
        self._fitted = False

    # ------------------------------------------------------------------
    # BaseDetector API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "GMMDetector":
        """
        Fit scaler + GMM, then compute the log-likelihood threshold.

        The threshold is the `contamination`-th percentile of the training
        log-likelihoods.  Samples scoring below this are flagged as outliers.
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)

        # Auto-compute decision threshold from training scores
        train_scores     = self.model.score_samples(X_scaled)  # log-likelihood per sample
        self._threshold  = float(
            np.percentile(train_scores, 100 * self.contamination)
        )
        self._fitted = True

        print(f"   GMM threshold (log-likelihood): {self._threshold:.4f}  "
              f"(contamination={self.contamination})")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns +1 (inlier) or -1 (outlier)."""
        self._check_fitted(self._fitted, "GMMDetector")
        scores = self.score(X)
        return np.where(scores >= self._threshold, 1, -1).astype(int)

    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Log-likelihood per sample.
        Higher (less negative) = more likely to be an inlier.
        """
        self._check_fitted(self._fitted, "GMMDetector")
        return self.model.score_samples(self.scaler.transform(X))

    # ------------------------------------------------------------------
    # Utility: component selection via BIC
    # ------------------------------------------------------------------

    @staticmethod
    def select_n_components(
        X: np.ndarray,
        max_components: int = 10,
        covariance_type: str = COVARIANCE,
        random_state: int = RANDOM_STATE,
    ) -> dict[int, float]:
        """
        Fit GMMs with n = 1 … max_components and return their BIC scores.
        Lower BIC = better model.  Pick the elbow.

        Example
        -------
            bic = GMMDetector.select_n_components(X)
            best_n = min(bic, key=bic.get)
        """
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        bic: dict[int, float] = {}

        for n in range(1, max_components + 1):
            gmm     = GaussianMixture(
                n_components    = n,
                covariance_type = covariance_type,
                random_state    = random_state,
                n_init          = 3,
            )
            gmm.fit(X_scaled)
            bic[n] = gmm.bic(X_scaled)
            print(f"   n_components={n:2d}  BIC={bic[n]:.1f}")

        return bic


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
        n_components:       int            = N_COMPONENTS,
        covariance_type:    str            = COVARIANCE,
        band_indices:       list[Bands]   | None = None,
        vegetation_indices: list[Indices] | None = None,
        rectangle:          bool           = False,
    ):
        self.detector = GMMDetector(
            n_components    = n_components,
            covariance_type = covariance_type,
        )
        self.band_indices       = band_indices
        self.vegetation_indices = vegetation_indices
        self.rectangle          = rectangle
        self.last_X: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Geometry helpers  (identical to ngrvi_pretrain.Pretrainer)
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

    def train(
        self,
        ortho_path:     Path,
        shapefile_path: Path,
        limit:          float = 0.8,
        select_components: bool = False,
    ):
        """
        Parameters
        ----------
        select_components : bool
            If True, run BIC selection (n=1…10) before fitting the final model
            and print a recommendation.  Useful for first-time tuning.
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
            bic  = GMMDetector.select_n_components(X)
            best = min(bic, key=bic.get)
            print(f"   → Recommended n_components = {best}  (lowest BIC)")
            # Re-create detector with the best n
            self.detector = GMMDetector(
                n_components    = best,
                covariance_type = self.detector.covariance_type,
            )

        print("\n── Fitting GMM ──────────────────────────────────────────")
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
        n_components       = N_COMPONENTS,
        covariance_type    = COVARIANCE,
        band_indices       = BANDS_TO_USE,
        vegetation_indices = INDICES_TO_USE,
        rectangle          = False,
    )

    for cfg in configs:
        trainer.train(
            ortho_path         = cfg.ortho_path,
            shapefile_path     = cfg.shapefile_path,
            limit              = 0.8,
            select_components  = False,  # set True on first run to pick best n
        )

    trainer.dump(OUTPUT_PATH / "gmm_model.joblib")
