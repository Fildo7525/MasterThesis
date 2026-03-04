"""
mahalanobis_inference.py

Drop-in inference wrapper that mirrors the exact pipeline used in
mahalanobis_analysis.py:

    raw pixel chip
        -> FeatureExtractor  (same as training)
        -> StandardScaler    (fitted on training data)
        -> PCA               (fitted on training data)
        -> Mahalanobis distance against saved mean / inv_covariance

Usage
-----
    infer = MahalanobisInference.from_saved(
        scaler_path  = "mahal_output/scaler.joblib",
        pca_path     = "mahal_output/pca.joblib",
        mean_path    = "mahal_output/mean.npy",
        inv_cov_path = "mahal_output/inv_covariance.npy",
        threshold    = json.load(open("mahal_output/thresholds.json"))["thresholds"]["95pct"],
    )

    result = infer.predict_polygon(geometry, rasterio_src)
    print(result.distance, result.is_inlier)

NOTE
----
  mahalanobis_analysis.py must be run with SAVE_TRANSFORMS = True (see below)
  so that scaler.joblib and pca.joblib are written alongside the .npy files.
  Add the two joblib.dump() calls shown at the bottom of this file to your
  analysis script if you have not already done so.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import rasterio
from shapely.geometry.base import BaseGeometry

from features.features import FeatureExtractor
from ngrvi_pretrain import Pretrainer
from ngrvi_approach import NgrviApproach


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MahalResult:
    distance:    float
    threshold:   float
    is_inlier:   bool
    feature_vec: np.ndarray   # raw (pre-scaler) feature vector

    def __repr__(self):
        status = "INLIER" if self.is_inlier else "OUTLIER"
        return (f"MahalResult({status}  d={self.distance:.4f}  "
                f"threshold={self.threshold:.4f})")


# ─────────────────────────────────────────────────────────────────────────────
# Inference class
# ─────────────────────────────────────────────────────────────────────────────

class MahalanobisInference:
    """
    Loads all saved training artefacts and exposes two prediction methods:

        predict_polygon(geometry, src)   — from a shapely geometry + open rasterio src
        predict_feature_vec(vec)         — from a raw feature vector (numpy array)
    """

    def __init__(
        self,
        scaler,                   # fitted sklearn StandardScaler
        pca,                      # fitted sklearn PCA  (or None if USE_PCA=False)
        mean:     np.ndarray,     # shape (n_dims,)
        inv_cov:  np.ndarray,     # shape (n_dims, n_dims)
        threshold: float,
        band_indices: list[int] | None = None,
        rectangle: bool = True,
    ):
        self.scaler      = scaler
        self.pca         = pca
        self.mean        = mean
        self.inv_cov     = inv_cov
        self.threshold   = threshold
        self.band_indices = band_indices
        self.rectangle   = rectangle

        # reuse the same feature extractor as training
        self._extractor = FeatureExtractor()

        # reuse the same NGRVI mask logic as training
        base_dir   = Path.home() / "SDU/MasterThesis"
        model_path = base_dir / "Orthomosaics/pretrain_output_model.joblib"
        self._ngrvi = NgrviApproach(model_path)

    # ── factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_saved(
        cls,
        scaler_path:  str | Path,
        pca_path:     str | Path | None,
        mean_path:    str | Path,
        inv_cov_path: str | Path,
        threshold:    float,
        band_indices: list[int] | None = None,
        rectangle:    bool = False,
    ) -> "MahalanobisInference":
        scaler  = joblib.load(scaler_path)
        pca     = joblib.load(pca_path) if pca_path else None
        mean    = np.load(mean_path)
        inv_cov = np.load(inv_cov_path)
        return cls(scaler, pca, mean, inv_cov, threshold, band_indices, rectangle)

    @classmethod
    def from_dir(
        cls,
        out_dir:       str | Path = "mahal_output",
        accept_pct:    str = "95pct",   # key in thresholds.json: "95pct" or "100pct"
        band_indices:  list[int] | None = None,
        rectangle:     bool = False,
    ) -> "MahalanobisInference":
        """Convenience loader — point at the output directory and go."""
        d    = Path(out_dir)
        meta = json.loads((d / "thresholds.json").read_text())
        pca_path = d / "pca.joblib"
        return cls.from_saved(
            scaler_path  = d / "scaler.joblib",
            pca_path     = pca_path if pca_path.exists() else None,
            mean_path    = d / "mean.npy",
            inv_cov_path = d / "inv_covariance.npy",
            threshold    = meta["thresholds"][accept_pct],
            band_indices = band_indices,
            rectangle    = rectangle,
        )

    # ── internal transform pipeline ──────────────────────────────────────────

    def _project(self, raw_vec: np.ndarray) -> np.ndarray:
        """
        Apply the same transforms used during training:
            StandardScaler -> PCA (optional)
        """
        x = self.scaler.transform(raw_vec.reshape(1, -1))   # (1, n_features)
        if self.pca is not None:
            x = self.pca.transform(x)                        # (1, n_dims)
        return x.ravel()                                     # (n_dims,)

    def _mahalanobis(self, projected: np.ndarray) -> float:
        diff = projected - self.mean
        return float(np.sqrt(max(diff @ self.inv_cov @ diff, 0.0)))

    # ── public API ───────────────────────────────────────────────────────────

    def predict_feature_vec(self, raw_vec: np.ndarray) -> MahalResult:
        """
        Score a pre-extracted feature vector (output of
        Pretrainer.extract_vector_from_polygon).

        Parameters
        ----------
        raw_vec : 1-D numpy array, shape (n_features,)
            The raw, un-scaled feature vector exactly as returned by the
            FeatureExtractor.

        Returns
        -------
        MahalResult
        """
        projected = self._project(raw_vec)
        dist      = self._mahalanobis(projected)
        return MahalResult(
            distance    = dist,
            threshold   = self.threshold,
            is_inlier   = dist <= self.threshold,
            feature_vec = raw_vec,
        )

    def predict_polygon(
        self,
        geometry: BaseGeometry,
        src:      rasterio.DatasetReader,
    ) -> Optional[MahalResult]:
        """
        End-to-end prediction directly from a shapely geometry.

        Parameters
        ----------
        geometry : shapely geometry
            The polygon / bounding box to score.
        src : rasterio.DatasetReader
            Open rasterio handle to the orthomosaic.

        Returns
        -------
        MahalResult, or None if the polygon is too small / empty.
        """
        # reuse identical feature extraction from Pretrainer
        raw_vec = _extract_vec(
            geometry     = geometry,
            src          = src,
            extractor    = self._extractor,
            ngrvi        = self._ngrvi,
            band_indices = self.band_indices,
            rectangle    = self.rectangle,
        )
        if raw_vec is None:
            return None

        return self.predict_feature_vec(raw_vec)

    def predict_batch(
        self,
        geometries: list[BaseGeometry],
        src:        rasterio.DatasetReader,
    ) -> list[Optional[MahalResult]]:
        """Score a list of geometries against the same open rasterio source."""
        return [self.predict_polygon(g, src) for g in geometries]

    def distance_only(self, raw_vec: np.ndarray) -> float:
        """Return just the scalar Mahalanobis distance (no threshold check)."""
        return self._mahalanobis(self._project(raw_vec))


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction helper (mirrors Pretrainer exactly)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_vec(
    geometry:     BaseGeometry,
    src:          rasterio.DatasetReader,
    extractor:    FeatureExtractor,
    ngrvi:        NgrviApproach,
    band_indices: list[int] | None,
    rectangle:    bool,
) -> np.ndarray | None:
    """
    Identical logic to Pretrainer.extract_vector_from_polygon — kept here so
    the inference path has zero dependency on the training class.
    """
    from rasterio.features import geometry_mask
    from rasterio.windows import from_bounds, Window, transform as win_transform

    minx, miny, maxx, maxy = geometry.bounds
    window = from_bounds(minx, miny, maxx, maxy, src.transform)
    window = window.round_lengths().round_offsets()
    window = window.intersection(Window(0, 0, src.width, src.height))

    wt         = win_transform(window, src.transform)
    win_h      = int(window.height)
    win_w      = int(window.width)
    pixel_mask = ~geometry_mask(
        [geometry], transform=wt, invert=False, out_shape=(win_h, win_w)
    )

    if pixel_mask.sum() < 9:
        return None

    if band_indices is None:
        # print(f"src.count: {src.count}")
        bands_1based  = list(range(1, 8))
        actual_indices = list(range(src.count-1))
    else:
        bands_1based  = [i + 1 for i in band_indices]
        actual_indices = band_indices

    chip  = src.read(bands_1based, window=window).astype(np.float32) / 65535
    # print(f"Read chip with shape {chip.shape} for geometry with bounds {geometry.bounds}")
    bands = [chip[i] for i in range(len(actual_indices))]

    ngrvi_mask, _ = ngrvi.create_ngrvi_mask(chip)

    results = extractor.process_multiband(
        bands,
        band_indices = list(range(chip.shape[0])),
        mask         = ngrvi_mask,
        rectangle    = rectangle,
    )

    values = []
    for band_name, feats in sorted(results.items()):
        if feats is None:
            continue
        for feat_name, val in sorted(feats.items()):
            values.append(float(val))

    return np.array(values) if values else None


# ─────────────────────────────────────────────────────────────────────────────
# Quick demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import geopandas as gpd
    from pathlib import Path
    from tqdm import tqdm

    HOME = Path.home()

    # ── load inference engine ─────────────────────────────────
    infer = MahalanobisInference.from_dir(
        out_dir    = "mahal_output",
        accept_pct = "95pct",       # use "100pct" to accept everything seen in training
    )
    print(f"Threshold: {infer.threshold:.4f}")

    # ── score a new orthomosaic + shapefile ───────────────────
    base_dir = Path.home() / "SDU/MasterThesis"
    ortho_path = base_dir / "Orthomosaics/20250827_Bjørnkjærvej_TestFlight_2_small.tif"
    shp_path   = base_dir / "OpenCV/report_results/output_20250827_Bjørnkjærvej_TestFlight_2_small/labels_shapefile.shp"

    gdf = gpd.read_file(shp_path)

    distances   = []
    labels      = []
    scored_idxs = []   # track original GDF row indices for the shapefile export

    with rasterio.open(ortho_path) as src:
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Scoring"):
            result = infer.predict_polygon(row.geometry, src)
            if result is None:
                continue
            distances.append(result.distance)
            labels.append(result.is_inlier)
            scored_idxs.append(idx)
            print(result)

    distances = np.array(distances)
    labels    = np.array(labels)
    print(f"\nInliers : {labels.sum()} / {len(labels)}")
    print(f"Outliers: {(~labels).sum()} / {len(labels)}")
    print(f"Distance range: [{distances.min():.4f}, {distances.max():.4f}]")

    # ── write inlier shapefile ────────────────────────────────────────────────
    # Attach scores back to the original GDF rows that were successfully scored,
    # then filter to inliers only and write out.
    scored_gdf = gdf.loc[scored_idxs].copy()
    scored_gdf["mahal_dist"] = distances          # useful for downstream inspection
    scored_gdf["is_inlier"]  = labels

    inlier_gdf = scored_gdf[scored_gdf["is_inlier"]].copy()
    inlier_gdf = inlier_gdf.drop(columns=["is_inlier"])  # clean up boolean column

    out_shp = shp_path.parent / "inliers_shapefile.shp"
    inlier_gdf.to_file(out_shp)
    print(f"\nInlier shapefile saved -> {out_shp}")
    print(f"  {len(inlier_gdf)} / {len(gdf)} polygons kept")
