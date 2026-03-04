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
# Multiprocessing support
# Each worker opens its OWN rasterio handle — sharing one handle across
# processes is not safe and causes corrupted reads.
# ─────────────────────────────────────────────────────────────────────────────

# Module-level worker state — populated once per worker via the initializer,
# so heavy objects (FeatureExtractor, NgrviApproach, scaler, PCA) are loaded
# only once per process instead of once per polygon.
_worker_infer      = None
_worker_ortho_path = None


def _worker_init(out_dir, accept_pct, ortho_path, band_indices, rectangle):
    """Runs once when a worker process starts."""
    global _worker_infer, _worker_ortho_path
    _worker_infer      = MahalanobisInference.from_dir(
        out_dir=out_dir, accept_pct=accept_pct,
        band_indices=band_indices, rectangle=rectangle,
    )
    _worker_ortho_path = ortho_path


def _worker_score(task):
    """
    Score a single geometry inside a worker process.

    task = (original_row_index, geometry_wkb_bytes)
    WKB is used because shapely geometries are not reliably picklable on all
    platforms when sent across process boundaries.

    Returns (idx, distance, is_inlier)  or  (idx, None, None) if too small.
    """
    from shapely import wkb as shapely_wkb
    idx, geom_wkb = task
    geometry = shapely_wkb.loads(geom_wkb)
    # Each worker opens its own file handle — no sharing, no corruption.
    with rasterio.open(_worker_ortho_path) as src:
        result = _worker_infer.predict_polygon(geometry, src)
    if result is None:
        return (idx, None, None)
    return (idx, result.distance, result.is_inlier)


def score_shapefile_parallel(
    ortho_path,
    shp_path,
    out_dir     = "mahal_output",
    accept_pct  = "95pct",
    band_indices = None,
    rectangle   = True,
    n_workers   = 4,
    chunk_size  = 8,
):
    """
    Score every polygon in a shapefile using a multiprocessing pool.

    Returns
    -------
    distances   : np.ndarray of float
    labels      : np.ndarray of bool  (True = inlier)
    scored_idxs : list of original GDF row indices
    gdf         : the (possibly reprojected) GeoDataFrame
    """
    import multiprocessing as mp
    from shapely import wkb as shapely_wkb
    import geopandas as gpd
    from tqdm import tqdm

    gdf = gpd.read_file(shp_path)
    with rasterio.open(ortho_path) as src:
        if gdf.crs != src.crs:
            print(f"  Reprojecting {gdf.crs} -> {src.crs}")
            gdf = gdf.to_crs(src.crs)

    # Serialise geometries to WKB so they can be pickled across processes
    tasks = [
        (idx, shapely_wkb.dumps(row.geometry))
        for idx, row in gdf.iterrows()
        if row.geometry is not None and not row.geometry.is_empty
    ]
    print(f"  Scoring {len(tasks)} polygons on {n_workers} workers ...")

    distances   = []
    labels      = []
    scored_idxs = []

    # "spawn" is safest: each worker is a clean Python process with no
    # inherited file handles or GDAL state from the parent.
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes   = n_workers,
        initializer = _worker_init,
        initargs    = (str(out_dir), accept_pct,
                       str(ortho_path), band_indices, rectangle),
    ) as pool:
        for idx, dist, is_inlier in tqdm(
            pool.imap(_worker_score, tasks, chunksize=chunk_size),
            total=len(tasks),
            desc="Scoring",
        ):
            if dist is None:
                continue
            distances.append(dist)
            labels.append(is_inlier)
            scored_idxs.append(idx)

    return np.array(distances), np.array(labels), scored_idxs, gdf


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import time
    from pathlib import Path

    HOME     = Path.home()
    base_dir = HOME / "SDU/MasterThesis"

    ORTHO_PATH = base_dir / "Orthomosaics/20250827_Bjørnkjærvej_TestFlight_2_small.tif"
    SHP_PATH   = base_dir / "OpenCV/report_results/output_20250827_Bjørnkjærvej_TestFlight_2_small/labels_shapefile.shp"
    OUT_DIR    = "mahal_output"
    ACCEPT_PCT = "95pct"

    cpu_count = os.cpu_count() or 1
    N_WORKERS  = max(1, cpu_count - 1)   # leave one core for the OS

    print(f"Using {N_WORKERS} worker processes")

    t0 = time.perf_counter()
    distances, labels, scored_idxs, gdf = score_shapefile_parallel(
        ortho_path   = ORTHO_PATH,
        shp_path     = SHP_PATH,
        out_dir      = OUT_DIR,
        accept_pct   = ACCEPT_PCT,
        n_workers    = N_WORKERS,
        rectangle    = False,
    )
    elapsed = time.perf_counter() - t0

    print(f"\nScored {len(distances)} polygons in {elapsed:.1f}s "
          f"({len(distances)/max(elapsed,1e-9):.1f} polygons/s)")
    print(f"Inliers : {labels.sum()} / {len(labels)}")
    print(f"Outliers: {(~labels).sum()} / {len(labels)}")
    if len(distances):
        print(f"Distance range: [{distances.min():.4f}, {distances.max():.4f}]")

    # ── write inlier shapefile ────────────────────────────────
    scored_gdf = gdf.loc[scored_idxs].copy()
    scored_gdf["mahal_dist"] = distances
    scored_gdf["is_inlier"]  = labels

    inlier_gdf = scored_gdf[scored_gdf["is_inlier"]].drop(columns=["is_inlier"])

    out_shp = SHP_PATH.parent / "inliers_shapefile.shp"
    inlier_gdf.to_file(out_shp)
    print(f"\nInlier shapefile saved -> {out_shp}")
    print(f"  {len(inlier_gdf)} / {len(gdf)} polygons kept")
