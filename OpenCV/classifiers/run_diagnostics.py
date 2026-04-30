"""
run_diagnostics.py
------------------
Diagnostic plots for any pretrainer whose detector follows the BaseDetector
convention (IsolationForestDetector or GMMDetector).

Public API
----------
    plot_feature_matrix(pretrainer, out_path, feature_names, max_features)
        Pairplot in original feature space — diagonal histograms, off-diagonal
        scatter, inliers vs outliers coloured.  Works for both detectors.

    plot_pca_importance(pretrainer, out_path, feature_names, max_loadings)
        Two-panel figure:
          Left  — scree bar chart: explained variance ratio per PC with a
                  cumulative variance line (elbow / Kaiser criterion aid).
          Right — stacked horizontal bar chart of the absolute loadings of
                  the top-N original features for each of the first few PCs
                  ("Plotr"-style importance chart).

    plot_pca_scatter(pretrainer, out_path)
        Scatter of every training sample projected onto PC1 and PC2, coloured
        by inlier/outlier, with the detector's decision boundary drawn as a
        filled contour over that 2-D slice.  Works for both detectors because
        scoring is routed through a thin PC-space wrapper.
"""

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from pathlib import Path

FONT_SIZE = 16

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_model(pretrainer):
    """
    Return the fitted detector object regardless of whether the pretrainer
    stores it as `.pipeline` (IForest) or `.detector` (old GMM style) or
    `.pipeline` (new GMM style).
    """
    model = getattr(pretrainer, "pipeline", None) or getattr(pretrainer, "detector", None)
    if model is None:
        raise ValueError("Pretrainer has neither .pipeline nor .detector attribute.")
    return model


def _resolve_pca(model):
    """
    Return the PCA object stored on the detector.
    Raises a clear error if the detector was not built with PCA.
    """
    pca = getattr(model, "pca_", None)
    if pca is None or not hasattr(pca, "components_"):
        raise ValueError(
            f"{type(model).__name__} does not expose a fitted .pca_ attribute. "
            "Make sure you are using the PCA-enabled detector from this release."
        )
    return pca


# ---------------------------------------------------------------------------
# 1. Feature matrix (pairplot in original space)
# ---------------------------------------------------------------------------

def plot_feature_matrix(pretrainer, out_path: Path,
                        feature_names=None, max_features: int = 10):
    """
    Pairplot-style matrix in the original (pre-PCA) feature space.

    Diagonal  : overlapping histograms (inliers / outliers)
    Off-diag  : scatter plots coloured by inlier / outlier

    Works for both IsolationForestDetector and GMMDetector because it only
    calls model.predict(X) on the raw feature matrix.

    Parameters
    ----------
    pretrainer    : trained Pretrainer from either pretrain script
    out_path      : Path where the PNG will be saved
    feature_names : list[str] | None — original feature names (pre-PCA)
    max_features  : int — cap the matrix size (it grows quadratically)
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    X = getattr(pretrainer, "last_X", None)
    if X is None:
        raise ValueError("pretrainer.last_X is None — call train() first.")

    model = _resolve_model(pretrainer)
    preds = model.predict(X)
    inliers  = preds == 1
    outliers = preds == -1

    n_features = min(max_features, X.shape[1])
    X_plot     = X[:, :n_features]
    names      = (feature_names or [f"F{i}" for i in range(X.shape[1])])[:n_features]

    INLIER_COLOR  = "#4C72B0"
    OUTLIER_COLOR = "#DD2152"

    fig, axes = plt.subplots(n_features, n_features,
                              figsize=(3 * n_features, 3 * n_features))
    fig.suptitle("Feature matrix (original space)", fontsize=FONT_SIZE*2+1, y=1.01)

    for i in range(n_features):
        for j in range(n_features):
            ax = axes[i, j]

            if i == j:
                ax.hist(X_plot[inliers,  i], bins=40, alpha=0.7, color=INLIER_COLOR,  label="Inliers")
                if outliers.sum() > 0:
                    ax.hist(X_plot[outliers, i], bins=40, alpha=0.7, color=OUTLIER_COLOR, label="Outliers")
            else:
                ax.scatter(X_plot[inliers,  j], X_plot[inliers,  i],
                           s=4, alpha=0.5, color=INLIER_COLOR,  label="Inliers")
                if outliers.sum() > 0:
                    ax.scatter(X_plot[outliers, j], X_plot[outliers, i],
                               s=4, alpha=0.8, color=OUTLIER_COLOR, label="Outliers")

            if i == n_features - 1:
                ax.set_xlabel(names[j], fontsize=FONT_SIZE*1.2)
            if j == 0:
                ax.set_ylabel(names[i], fontsize=FONT_SIZE*1.2)
            ax.tick_params(axis="both", which="both", labelsize=5)

    handles = [
        mpatches.Patch(color=INLIER_COLOR,  label="Inliers"),
        mpatches.Patch(color=OUTLIER_COLOR, label="Outliers"),
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=FONT_SIZE*1.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Feature matrix saved → {out_path}")


# ---------------------------------------------------------------------------
# 2. PCA importance (scree + loadings)
# ---------------------------------------------------------------------------

def plot_pca_importance(pretrainer, out_path: Path,
                        feature_names=None, max_loadings: int = 10,
                        max_pcs_in_loading: int = 5,
                        pca_variance:float = 95):
    """
    Two-panel PCA importance figure.

    Left panel — Scree chart
        Bar chart of explained variance ratio per principal component with a
        cumulative variance line.  A dashed horizontal line marks 95 % to
        help identify the knee of the curve.

    Right panel — Feature loading chart  ("Piotr / plotr" style)
        Horizontal stacked bar chart.  Each row is one of the top original
        features (ranked by their maximum absolute loading across the first
        `max_pcs_in_loading` PCs).  Bars are split by PC contribution so
        you can see which components each feature drives.

    Parameters
    ----------
    pretrainer          : trained Pretrainer
    out_path            : Path where the PNG will be saved
    feature_names       : list[str] | None — original feature names
    max_loadings        : int — how many features to show in the loading chart
    max_pcs_in_loading  : int — how many PCs to include in the loading chart
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = _resolve_model(pretrainer)
    pca   = _resolve_pca(model)

    evr        = pca.explained_variance_ratio_          # (n_components,)
    components = pca.components_                         # (n_components, n_features)
    n_pcs      = len(evr)
    n_features = components.shape[1]
    names      = (feature_names or [f"F{i}" for i in range(n_features)])

    # ── Left: scree ─────────────────────────────────────────────────────────
    fig, (ax_scree, ax_load) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("PCA component analysis", fontsize=FONT_SIZE+1)

    pc_labels = [f"PC{i+1}" for i in range(n_pcs)]
    cumvar    = np.cumsum(evr)

    bars = ax_scree.bar(pc_labels, evr * 100, color="#4C72B0", alpha=0.85,
                         zorder=2, label="Explained variance")
    ax_scree.plot(pc_labels, cumvar * 100, color="#DD8452", marker="o",
                  linewidth=2, markersize=5, zorder=3, label="Cumulative")
    ax_scree.axhline(pca_variance*100, color="grey", linestyle="--", linewidth=1, label="95 % threshold")

    ax_scree.set_xlabel("Principal component", fontsize=FONT_SIZE)
    ax_scree.set_ylabel("Explained variance (%)", fontsize=FONT_SIZE)
    ax_scree.set_title("Scree chart", fontsize=FONT_SIZE+1)
    ax_scree.set_ylim(0, 105)
    ax_scree.legend(fontsize=FONT_SIZE-2)
    ax_scree.tick_params(axis="x", rotation=45)

    # Annotate each bar with its percentage
    for bar, val in zip(bars, evr):
        ax_scree.text(bar.get_x() + bar.get_width() / 2,
                      bar.get_height() + 0.8,
                      f"{val*100:.1f}%", ha="center", va="bottom", fontsize=FONT_SIZE-1)

    # ── Right: loading chart ─────────────────────────────────────────────────
    n_pcs_plot = min(max_pcs_in_loading, n_pcs)

    # Rank features by their maximum absolute loading across the first n_pcs_plot PCs
    abs_loadings = np.abs(components[:n_pcs_plot])          # (n_pcs_plot, n_features)
    max_per_feat = abs_loadings.max(axis=0)                 # (n_features,)
    top_idx      = np.argsort(max_per_feat)[::-1][:max_loadings]  # indices, descending

    top_names    = [names[i] for i in top_idx]
    top_loadings = abs_loadings[:, top_idx]                 # (n_pcs_plot, max_loadings)

    # Colour palette for PCs
    pc_colors = plt.cm.tab10(np.linspace(0, 0.6, n_pcs_plot))

    # Stacked horizontal bar — each PC adds its slice
    y_pos     = np.arange(max_loadings)
    lefts     = np.zeros(max_loadings)

    for pc_i in range(n_pcs_plot):
        widths = top_loadings[pc_i]
        ax_load.barh(y_pos, widths, left=lefts, height=0.6,
                     color=pc_colors[pc_i], alpha=0.85, label=f"PC{pc_i+1}")
        ax_load.tick_params(labelsize=FONT_SIZE-1)
        lefts += widths

    ax_load.set_yticks(y_pos)
    ax_load.set_yticklabels(top_names, fontsize=FONT_SIZE)
    ax_load.invert_yaxis()   # most important at the top
    ax_load.set_xlabel("Summed |loading| across selected PCs", fontsize=FONT_SIZE-1)
    ax_load.set_title(f"Top-{max_loadings} feature loadings (first {n_pcs_plot} PCs)", fontsize=FONT_SIZE+1)
    ax_load.legend(fontsize=FONT_SIZE-1, loc="lower right")
    ax_load.axvline(0, color="black", linewidth=0.8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"PCA importance saved → {out_path}")


# ---------------------------------------------------------------------------
# 3. PCA scatter with decision boundary
# ---------------------------------------------------------------------------

def plot_pca_scatter(pretrainer, out_path: Path, resolution: int = 300):
    """
    Scatter of training samples projected onto PC1 and PC2, coloured by
    inlier / outlier label, overlaid with the detector's decision boundary.

    The decision boundary is drawn by evaluating the detector's score on a
    fine grid in (PC1, PC2) space (all other PCs held at zero).  A filled
    contour shows the score landscape; the boundary itself (score = threshold
    for GMM, score = 0 for IForest) is drawn as a solid black line.

    Parameters
    ----------
    pretrainer : trained Pretrainer
    out_path   : Path where the PNG will be saved
    resolution : int — grid resolution (higher = smoother boundary, slower)
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    X = getattr(pretrainer, "last_X", None)
    if X is None:
        raise ValueError("pretrainer.last_X is None — call train() first.")

    model = _resolve_model(pretrainer)
    pca   = _resolve_pca(model)

    # ── Project training data to PCA space ──────────────────────────────────
    X_scaled = model.scaler_.transform(X)
    X_pca    = pca.transform(X_scaled)          # (n_samples, n_components)

    preds    = model.predict(X)
    inliers  = preds == 1
    outliers = preds == -1

    pc1 = X_pca[:, 0]
    pc2 = X_pca[:, 1]

    # ── Build grid in PC1–PC2 plane ──────────────────────────────────────────
    margin = 0.15   # fraction of range to add on each side
    x_min, x_max = pc1.min(), pc1.max()
    y_min, y_max = pc2.min(), pc2.max()
    dx = (x_max - x_min) * margin
    dy = (y_max - y_min) * margin

    xx, yy = np.meshgrid(
        np.linspace(x_min - dx, x_max + dx, resolution),
        np.linspace(y_min - dy, y_max + dy, resolution),
    )

    # Grid points in PCA space — all remaining PCs set to zero
    n_components = X_pca.shape[1]
    grid_pca     = np.zeros((xx.size, n_components))
    grid_pca[:, 0] = xx.ravel()
    grid_pca[:, 1] = yy.ravel()

    # Score the grid directly through the underlying model in PCA space
    # (bypassing the scaler/PCA transform since we're already in PCA space)
    grid_scores = model.model.score_samples(grid_pca).reshape(xx.shape)

    # Determine decision threshold — GMM stores _threshold, IForest uses 0
    threshold = getattr(model, "_threshold", 0.0)

    # ── Plot ─────────────────────────────────────────────────────────────────
    INLIER_COLOR  = "#4C72B0"
    OUTLIER_COLOR = "#DD8452"

    fig, ax = plt.subplots(figsize=(8, 6))

    # Score landscape as filled contour
    cf = ax.contourf(xx, yy, grid_scores, levels=30,
                     cmap="RdYlGn", alpha=0.45)

    cb = plt.colorbar(cf, ax=ax)
    cb.set_label(label="Anomaly score (higher = more normal)",size=FONT_SIZE-1)

    # Decision boundary
    ax.contour(xx, yy, grid_scores, levels=[threshold],
               colors="black", linewidths=1.5, linestyles="--")

    # Training points
    ax.scatter(pc1[inliers],  pc2[inliers],  s=18, color=INLIER_COLOR,
               alpha=0.7, label=f"Inliers ({inliers.sum()})", zorder=3)
    if outliers.sum() > 0:
        ax.scatter(pc1[outliers], pc2[outliers], s=18, color=OUTLIER_COLOR,
                   alpha=0.9, marker="x", label=f"Outliers ({outliers.sum()})", zorder=4)

    evr = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1  ({evr[0]*100:.1f} % variance)", fontsize=FONT_SIZE)
    ax.set_ylabel(f"PC2  ({evr[1]*100:.1f} % variance)", fontsize=FONT_SIZE)
    title = f"PC1 vs PC2 — {type(model).__name__} decision boundary\n"
    if threshold != 0.0:
        title += f"(dashed line = threshold {threshold:.3f})"

    ax.set_title(title, fontsize=FONT_SIZE+1)
    ax.legend(fontsize=FONT_SIZE-1)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"PCA scatter saved → {out_path}")
