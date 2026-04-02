"""
mahalanobis_analysis.py

1. Extract features via Pretrainer
2. Standardise + PCA (optional blackning)
3. Compute mean vector and covariance matrix of the training distribution
4. Compute Mahalanobis distance for every training sample
5. Find the threshold that accepts 95 % (and 100 %) of training samples
6. Save: mean.npy, covariance.npy, inv_covariance.npy, thresholds.json
7. Plot:
   a) Mahalanobis distance distribution + threshold lines
   b) Chi-squared QQ-plot (theoretical vs empirical)
   c) 2-D PCA scatter coloured by Mahalanobis distance

Dependencies:
    pip install numpy scipy matplotlib seaborn scikit-learn joblib

Usage:
    python mahalanobis_analysis.py
"""

from __future__ import annotations
import json
import joblib
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf   # robust cov estimator
from scipy.stats import shapiro

from create_indexes import Indices, Bands
import shutil

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
HOME = Path.home()

CONFIGS = [
    dict(
        ortho  = HOME / "SDU/MasterThesis/Orthomosaics/20250827_Bjørnkjærvej_TestFlight_2_small.tif",
        shapes = HOME / "SDU/MasterThesis/Orthomosaics/shapefiles/small/small_obb_test.shp",
        limit = 0.8,
    ),
    dict(
        ortho  = HOME / "SDU/MasterThesis/Orthomosaics/20250827_Bjørnkjærvej_TestFlight_2_mid.tif",
        shapes = HOME / "SDU/MasterThesis/Orthomosaics/shapefiles/mid/mid_obb_test.shp",
        limit = 0.8,
    ),
    dict(
        ortho = HOME / "SDU/MasterThesis/Orthomosaics/20250827_Bjørnkjærvej_TestFlight_2_bigger_v2.tif",
        shapes = HOME / "SDU/MasterThesis/Orthomosaics/shapefiles/large/large_obb_test.shp",
        limit = 0.8,
    )
]

RANDOM_SEED      = 42
OUT_DIR          = Path("./mahal_output_INDICES_v5")
NU               = 0.05
ACCEPT_FRACTIONS = [0.95, 0.975, 0.99, 1.00]   # threshold percentiles to compute & plot
USE_PCA          = True            # reduce to PCA space before computing cov
PCA_VARIANCE     = 0.99            # keep enough PCs to explain this fraction
USE_LEDOIT_WOLF  = True            # robust covariance estimator (better for small N)
BANDS_TO_USE     = [ Bands.RED, Bands.GREEN, Bands.BLUE ]            # None means all otherwise a list of Bands should be supplied.
INDICES_TO_USE   = None # [ Indices.NGRDI, ]            # None means all otherwise a list of Indices should be supplied.
# ──────────────────────────────────────────────────────────────

np.random.seed(RANDOM_SEED)


# ── data helpers ──────────────────────────────────────────────

def build_X(configs):
    from ngrvi_pretrain import Pretrainer
    trainer = Pretrainer(nu=NU, kernel="rbf", band_indices=None, rectangle=True)
    rows = []
    for cfg in configs:
        X = trainer.build_feature_matrix(
            ortho_path=Path(cfg["ortho"]),
            shapefile_path=Path(cfg["shapes"]),
            band_indices=[band for band in BANDS_TO_USE] if BANDS_TO_USE is not None else None,
            vegetation_indices = INDICES_TO_USE,
            limit=cfg["limit"],
        )
        rows.append(X)
    return np.vstack(rows)


def clean(X: np.ndarray) -> np.ndarray:
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


# ── Mahalanobis helpers ───────────────────────────────────────

def mahalanobis_distance(X: np.ndarray,
                          mean: np.ndarray,
                          inv_cov: np.ndarray) -> np.ndarray:
    """Vectorised Mahalanobis distance for every row of X."""
    diff = X - mean                          # (n, d)
    left = diff @ inv_cov                    # (n, d)
    sq   = np.einsum("ij,ij->i", left, diff) # (n,)
    return np.sqrt(np.maximum(sq, 0.0))


# ── plotting helpers ──────────────────────────────────────────

BG, GRID = "#ffffff", "#5c5c5c"
ACC, AMB, RED, GRN = "#0986be", "#997300", "#ef5350", "#69f0ae"
PRP = "#ab47bc"
ORG = "#ff7043"

THRESH_COLORS = {0.95: AMB, 0.975: PRP, 0.99: ORG, 1.00: RED}
THRESH_LABELS = {0.95: "95 %", 0.975: "97.5 %", 0.99: "99 %", 1.00: "100 %"}


def style_ax(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors="black", labelsize=9)
    ax.spines[:].set_color(GRID)
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    ax.title.set_color("black")


def plot_distance_distribution(distances: np.ndarray,
                                thresholds: dict[float, float],
                                n_dims: int,
                                out_path: Path) -> None:
    """
    Histogram of Mahalanobis distances with:
      - KDE overlay
      - empirical threshold lines (95 %, 100 %)
      - theoretical chi distribution (sqrt of chi-squared with n_dims dof)
    """
    fig, ax = plt.subplots(figsize=(11, 5), facecolor=BG)
    style_ax(ax)

    # histogram
    ax.hist(distances, bins=40, density=True,
            color="#1c3a4a", edgecolor="#2a5a70", linewidth=0.5,
            label="Empirical", zorder=2)

    # KDE
    kde_x = np.linspace(0, distances.max() * 1.05, 400)
    kde   = stats.gaussian_kde(distances)
    ax.plot(kde_x, kde(kde_x), color=ACC, lw=2.2, zorder=3, label="KDE")

    # theoretical chi distribution (Mahalanobis ~ chi(d) under Gaussian assumption)
    chi_pdf = stats.chi.pdf(kde_x, df=n_dims)
    ax.plot(kde_x, chi_pdf, color=GRN, lw=1.8, ls="--", zorder=3,
            label=f"Chi(df={n_dims}) theoretical")

    # threshold lines
    for frac, thresh in thresholds.items():
        col = THRESH_COLORS[frac]
        ax.axvline(thresh, color=col, lw=2, ls="--", zorder=4,
                   label=f"Threshold {THRESH_LABELS[frac]}  (d={thresh:.3f})")
        ax.text(thresh + distances.max() * 0.005,
                ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 0.01,
                f"{thresh:.3f}", color=col, fontsize=9, va="top")

    ax.set_xlabel("Mahalanobis Distance")
    ax.set_ylabel("Density")
    ax.set_title("Training-Set Mahalanobis Distance Distribution")
    ax.legend(fontsize=9, labelcolor="black", facecolor=GRID, edgecolor="none")
    ax.grid(color=GRID, zorder=0, alpha=0.6)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Distance distribution saved -> {out_path}")


def plot_qq(distances: np.ndarray, n_dims: int, out_path: Path) -> None:
    """Chi QQ-plot: empirical quantiles vs theoretical chi(d) quantiles."""
    n      = len(distances)
    probs  = (np.arange(1, n + 1) - 0.5) / n
    emp_q  = np.sort(distances)
    theo_q = stats.chi.ppf(probs, df=n_dims)

    fig, ax = plt.subplots(figsize=(6, 6), facecolor=BG)
    style_ax(ax)

    ax.scatter(theo_q, emp_q, s=18, alpha=0.6, color=ACC, edgecolor="none", zorder=3)

    # 45-degree reference line
    lo = min(theo_q.min(), emp_q.min())
    hi = max(theo_q.max(), emp_q.max())
    ax.plot([lo, hi], [lo, hi], color=GRN, lw=1.8, ls="--", zorder=4,
            label="Perfect Gaussian fit")

    ax.set_xlabel(f"Theoretical Chi(df={n_dims}) quantiles")
    ax.set_ylabel("Empirical Mahalanobis distance quantiles")
    ax.set_title("Chi QQ-Plot\n(deviation from line = non-Gaussianity)")
    ax.legend(fontsize=9, labelcolor="black", facecolor=GRID, edgecolor="none")
    ax.grid(color=GRID, zorder=0, alpha=0.6)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  QQ-plot saved -> {out_path}")


def plot_pca_scatter(X_pca: np.ndarray,
                     distances: np.ndarray,
                     thresholds: dict[float, float],
                     out_path: Path,
                     pca: PCA) -> None:
    """
    2-D PCA scatter coloured by Mahalanobis distance.
    Dashed ellipses mark the 95 % and 100 % acceptance boundaries
    (drawn in PC1-PC2 space as iso-distance contours).
    """
    x, y = X_pca[:, 0], X_pca[:, 1]
    ev   = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(9, 7), facecolor=BG)
    style_ax(ax)

    sc = ax.scatter(x, y, c=distances, cmap="plasma",
                    s=40, alpha=0.85, edgecolor="none", zorder=3)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label("Mahalanobis Distance", color="black")
    cbar.ax.yaxis.set_tick_params(color="black", labelsize=8, labelcolor="black")

    # draw ellipses at each threshold (based on std of PC1 & PC2)
    theta  = np.linspace(0, 2 * np.pi, 300)
    # mean and std of distances projected back to PC space
    # we approximate the iso-contour as an ellipse scaled by threshold / mean_dist
    mean_dist = distances.mean()
    for frac, thresh in thresholds.items():
        scale = thresh / mean_dist if mean_dist > 0 else 1.0
        rx    = x.std() * scale * 2.2
        ry    = y.std() * scale * 2.2
        col   = THRESH_COLORS[frac]
        ax.plot(rx * np.cos(theta),
                ry * np.sin(theta),
                color=col, lw=1.8, ls="--", zorder=4,
                label=f"Approx boundary {THRESH_LABELS[frac]}  (d={thresh:.3f})")

    ax.set_xlabel(f"PC1  ({ev[0]:.1f} %)", fontsize=10)
    ax.set_ylabel(f"PC2  ({ev[1]:.1f} %)", fontsize=10)
    ax.set_title("PCA Scatter — coloured by Mahalanobis Distance", fontsize=12)
    ax.legend(fontsize=9, labelcolor="black", facecolor=GRID, edgecolor="none")
    ax.grid(color=GRID, zorder=0, alpha=0.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  PCA scatter saved -> {out_path}")


def plot_cumulative_acceptance(distances: np.ndarray,
                                thresholds: dict[float, float],
                                out_path: Path) -> None:
    """ECDF of Mahalanobis distances — shows what fraction is accepted at each threshold."""
    sorted_d = np.sort(distances)
    ecdf     = np.arange(1, len(sorted_d) + 1) / len(sorted_d)

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
    style_ax(ax)

    ax.plot(sorted_d, ecdf * 100, color=ACC, lw=2.5, zorder=3, label="ECDF")
    ax.fill_between(sorted_d, ecdf * 100, alpha=0.12, color=ACC)

    for frac, thresh in thresholds.items():
        col = THRESH_COLORS[frac]
        ax.axvline(thresh, color=col, lw=2, ls="--", zorder=4)
        ax.axhline(frac * 100, color=col, lw=1.2, ls=":", zorder=4,
                   label=f"d={thresh:.3f}  accepts {frac*100:.0f}%")

    ax.set_xlabel("Mahalanobis Distance Threshold")
    ax.set_ylabel("% of Training Samples Accepted")
    ax.set_title("Cumulative Acceptance — choose your threshold")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9, labelcolor="black", facecolor=GRID, edgecolor="none")
    ax.grid(color=GRID, zorder=0, alpha=0.6)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Cumulative acceptance saved -> {out_path}")


# ── main ───────────────────────────────────────────────────────

def main():
    if OUT_DIR.exists():
        existing_dirs = OUT_DIR.parent.glob(f"{OUT_DIR.stem}_*")
        new_name_for_existing = OUT_DIR.parent / f"{OUT_DIR.stem}_0001"
        nums = sorted(list(map(int, (p.stem.split("_")[-1] for p in existing_dirs if p.is_dir() and p.stem.startswith(OUT_DIR.stem)))))
        max_num = nums[-1] if nums else 0
        new_name_for_existing = OUT_DIR.parent / f"{OUT_DIR.stem}_{max_num + 1:04}"

        shutil.move(str(OUT_DIR), str(new_name_for_existing))
        print(f"The old output is moved to {new_name_for_existing}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. features
    print("-- Extracting features ---------------------------------------")
    X_raw = clean(build_X(CONFIGS))
    print(f"   Raw shape: {X_raw.shape}")

    # 2. standardise
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # 3. optional PCA whitening
    if USE_PCA:
        pca_full = PCA(n_components=min(X_scaled.shape), random_state=42)
        pca_full.fit(X_scaled)
        cumvar   = np.cumsum(pca_full.explained_variance_ratio_)
        n_keep   = int(np.searchsorted(cumvar, PCA_VARIANCE)) + 1
        n_keep   = max(2, n_keep)
        pca      = PCA(n_components=n_keep, random_state=42)
        X_work   = pca.fit_transform(X_scaled)
        print(f"   PCA: keeping {n_keep} components "
              f"({cumvar[n_keep-1]*100:.1f}% variance)")
    else:
        pca    = PCA(n_components=2, random_state=42).fit(X_scaled)   # for scatter only
        X_work = X_scaled
        n_keep = X_scaled.shape[1]

    n_dims = X_work.shape[1]

    # 4. mean + covariance
    print("\n-- Computing distribution parameters -------------------------")
    mean_vec = X_work.mean(axis=0)

    if USE_LEDOIT_WOLF:
        lw    = LedoitWolf().fit(X_work)
        cov   = lw.covariance_
        print(f"   Ledoit-Wolf shrinkage: {lw.shrinkage_:.4f}")
    else:
        cov = np.cov(X_work, rowvar=False)

    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov)
        print("   Warning: covariance singular — using pseudo-inverse")

    print(f"   Mean vector shape  : {mean_vec.shape}")
    print(f"   Covariance shape   : {cov.shape}")

    stat, p = shapiro(X_work)
    print(f"Test done with {X_work.shape[0]} samples and {X_work.shape[1]} dimensions.")
    if p < 0.05:
        print(f"\n   Shapiro-Wilk test: stat={stat:.4f}, p={p:.4e} -> reject normality")
    else:
        print(f"\n   Shapiro-Wilk test: stat={stat:.4f}, p={p:.4e} -> fail to reject normality")

    # 5. Mahalanobis distances
    distances = mahalanobis_distance(X_work, mean_vec, inv_cov)
    print(f"\n   Distance stats:")
    print(f"     min  = {distances.min():.4f}")
    print(f"     mean = {distances.mean():.4f}")
    print(f"     max  = {distances.max():.4f}")
    print(f"     std  = {distances.std():.4f}")

    # 6. thresholds
    thresholds = {}
    print("\n   Acceptance thresholds:")
    for frac in ACCEPT_FRACTIONS:
        t = float(np.percentile(distances, frac * 100))
        thresholds[frac] = t
        print(f"     {frac*100:5.1f}%  ->  d = {t:.5f}")

    print("")
    for frac in ACCEPT_FRACTIONS:
        chi_95 = stats.chi.ppf(frac, df=n_dims)
        print(f"   Theoretical chi(df={n_dims}, p={frac}) = {chi_95:.5f}")

    # 7. save artefacts
    np.save(OUT_DIR / "mean.npy",         mean_vec)
    np.save(OUT_DIR / "covariance.npy",   cov)
    np.save(OUT_DIR / "inv_covariance.npy", inv_cov)

    meta = {
        "n_samples":          int(X_work.shape[0]),
        "n_dims":             int(n_dims),
        "use_pca":            USE_PCA,
        "pca_variance_kept":  float(PCA_VARIANCE) if USE_PCA else None,
        "use_ledoit_wolf":    USE_LEDOIT_WOLF,
        "distance_stats": {
            "min":  float(distances.min()),
            "mean": float(distances.mean()),
            "max":  float(distances.max()),
            "std":  float(distances.std()),
        },
        "thresholds": {f"{k*100:.2f}pct": v for k, v in thresholds.items()},
        "chi_theoretical_95": float(stats.chi.ppf(0.95, df=n_dims)),
        "bands": BANDS_TO_USE or "None",
        "indices": INDICES_TO_USE or "None",
    }
    with open(OUT_DIR / "thresholds.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n   Saved: mean.npy, covariance.npy, inv_covariance.npy, thresholds.json")

    # 8. plots
    print("\n-- Plotting --------------------------------------------------")
    X_pca2 = pca.transform(X_scaled) if USE_PCA else pca.fit_transform(X_scaled)

    plot_distance_distribution(distances, thresholds, n_dims,
                                OUT_DIR / "distance_distribution.png")
    plot_qq(distances, n_dims,
            OUT_DIR / "chi_qqplot.png")
    plot_pca_scatter(X_pca2, distances, thresholds,
                     OUT_DIR / "pca_scatter.png", pca)
    plot_cumulative_acceptance(distances, thresholds,
                                OUT_DIR / "cumulative_acceptance.png")

    joblib.dump(pca, OUT_DIR / "pca.joblib")
    joblib.dump(scaler, OUT_DIR / "scaler.joblib")
    print(f"\nDone — all outputs in: {OUT_DIR.resolve()}")
    print("\n-- How to use at inference time ------------------------------")
    print("""
  import numpy as np, json

  mean    = np.load('mahal_output/mean.npy')
  inv_cov = np.load('mahal_output/inv_covariance.npy')
  meta    = json.load(open('mahal_output/thresholds.json'))
  thresh  = meta['thresholds']['95pct']   # or '100pct'

  def mahal(x, mean, inv_cov):
      d = x - mean
      return float(np.sqrt(d @ inv_cov @ d))

  # for a new sample (already scaled + PCA projected):
  dist = mahal(new_sample_projected, mean, inv_cov)
  is_inlier = dist <= thresh
""")


if __name__ == "__main__":
    main()
