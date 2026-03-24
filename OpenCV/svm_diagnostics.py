from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


def _decision_scores(pipeline: Pipeline, X: np.ndarray) -> np.ndarray:
    """Return raw decision-function scores (positive = inlier)."""
    return pipeline.decision_function(X)


def _predictions(pipeline: Pipeline, X: np.ndarray) -> np.ndarray:
    """Return +1 (inlier) / -1 (outlier) predictions."""
    return pipeline.predict(X)


# ── 1. Decision score histogram ──────────────────────────────────────────────

def plot_decision_histogram(pipeline: Pipeline, X: np.ndarray, ax: plt.Axes | None = None,
                            save_path: Path | None = None):
    """
    Histogram of decision-function scores.
    Inliers (score > 0) are green, outliers (score < 0) are red.
    The vertical dashed line marks the decision boundary at 0.
    """
    scores = _decision_scores(pipeline, X)
    preds  = _predictions(pipeline, X)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(scores[preds ==  1], bins=40, color="#4caf50", alpha=0.7, label=f"Inlier  (n={( preds== 1).sum()})")
    ax.hist(scores[preds == -1], bins=40, color="#f44336", alpha=0.7, label=f"Outlier (n={(preds==-1).sum()})")
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2, label="Decision boundary")
    ax.set_xlabel("Decision function score")
    ax.set_ylabel("Count")
    ax.set_title("Decision score distribution")
    ax.legend()

    if standalone:
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"  Saved → {save_path}")
        plt.show()


# ── 2. PCA 2-D projection ────────────────────────────────────────────────────

def plot_pca_projection(pipeline: Pipeline, X: np.ndarray, ax: plt.Axes | None = None,
                        save_path: Path | None = None):
    """
    Project the (scaled) feature space to 2 principal components and
    colour each point by inlier / outlier status.
    Decision boundary contour is drawn by scoring a grid in PCA space.
    """
    scaler = pipeline.named_steps["scaler"]
    svm    = pipeline.named_steps["oc_svm"]
    X_scaled = scaler.transform(X)

    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)
    preds = _predictions(pipeline, X)

    # Build a grid in 2D PCA space to draw the decision boundary
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # Project grid back to full PCA space (remaining components = 0)
    n_components = min(pca.n_components_, X_scaled.shape[1])
    grid_pca = np.c_[xx.ravel(), yy.ravel()]
    pad = np.zeros((grid_pca.shape[0], X_scaled.shape[1] - 2))
    grid_full = np.hstack([grid_pca, pad])
    grid_orig = pca.inverse_transform(grid_full) if hasattr(pca, "inverse_transform") else grid_full
    Z = svm.decision_function(grid_orig).reshape(xx.shape)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 6))

    ax.contourf(xx, yy, Z, levels=[-np.inf, 0], colors=["#ffcdd2"], alpha=0.4)
    ax.contour( xx, yy, Z, levels=[0], colors=["#d32f2f"], linewidths=1.5)

    colors = np.where(preds == 1, "#4caf50", "#f44336")
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, s=12, alpha=0.6, edgecolors="none")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4caf50", label=f"Inlier  (n={(preds== 1).sum()})"),
        Patch(facecolor="#f44336", label=f"Outlier (n={(preds==-1).sum()})"),
    ]
    ax.legend(handles=legend_elements)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title("PCA projection with decision boundary")

    if standalone:
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"  Saved → {save_path}")
        plt.show()


# ── 3. Nu sweep ──────────────────────────────────────────────────────────────

def plot_nu_sweep(pipeline: Pipeline, X: np.ndarray, ax: plt.Axes | None = None,
                  save_path: Path | None = None,
                  nu_values: list[float] | None = None):
    """
    Re-fit OneClassSVM for a range of nu values (keeping everything else fixed)
    and plot the fraction of training samples classified as inliers.
    Helps pick a nu that engulfs a desired percentage of training data.
    """
    from sklearn.svm import OneClassSVM
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline as SKPipeline

    if nu_values is None:
        nu_values = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

    orig_svm = pipeline.named_steps["oc_svm"]
    kernel   = orig_svm.kernel

    inlier_fracs = []
    for nu in nu_values:
        p = SKPipeline([
            ("scaler", StandardScaler()),
            ("oc_svm", OneClassSVM(kernel=kernel, nu=nu, gamma="scale")),
        ])
        p.fit(X)
        preds = p.predict(X)
        inlier_fracs.append((preds == 1).mean())

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(nu_values, [f * 100 for f in inlier_fracs], "o-", color="#1976d2", linewidth=2)
    ax.axhline(95, color="grey", linestyle="--", linewidth=1, label="95 % threshold")
    ax.set_xlabel("nu")
    ax.set_ylabel("Inlier % on training data")
    ax.set_title("Inlier fraction vs nu  (lower nu → more inclusive)")
    ax.set_xticks(nu_values)
    ax.set_xticklabels([str(v) for v in nu_values], rotation=45)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    if standalone:
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"  Saved → {save_path}")
        plt.show()

    return list(zip(nu_values, inlier_fracs))


# ── 4. Permutation feature importance ────────────────────────────────────────

def plot_feature_importance(pipeline: Pipeline, X: np.ndarray,
                            feature_names: list[str] | None = None,
                            top_n: int = 20,
                            ax: plt.Axes | None = None,
                            save_path: Path | None = None):
    """
    Proxy feature importance via permutation: shuffle each feature column,
    measure drop in mean decision score, then restore.
    Higher drop → feature matters more.
    """
    baseline = _decision_scores(pipeline, X).mean()
    importances = []

    rng = np.random.default_rng(42)
    for col in range(X.shape[1]):
        X_perm = X.copy()
        X_perm[:, col] = rng.permutation(X_perm[:, col])
        drop = baseline - _decision_scores(pipeline, X_perm).mean()
        importances.append(drop)

    importances = np.array(importances)
    order = np.argsort(importances)[::-1][:top_n]

    if feature_names is None:
        feature_names = [f"feat_{i}" for i in range(X.shape[1])]
    names_sorted = [feature_names[i] for i in order]
    vals_sorted  = importances[order]

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.35)))

    colors = ["#ef5350" if v > 0 else "#90a4ae" for v in vals_sorted]
    ax.barh(names_sorted[::-1], vals_sorted[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean score drop after permutation")
    ax.set_title(f"Top-{top_n} feature importance (permutation)")

    if standalone:
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"  Saved → {save_path}")
        plt.show()


# ── 5. Combined dashboard ────────────────────────────────────────────────────

def plot_all(pipeline: Pipeline, X: np.ndarray,
             feature_names: list[str] | None = None,
             out_dir: Path | None = None):
    """
    Render all four diagnostic plots in a single figure and optionally save each.
    """
    out_dir = Path(out_dir) if out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax_hist  = fig.add_subplot(gs[0, 0])
    ax_pca   = fig.add_subplot(gs[0, 1])
    ax_nu    = fig.add_subplot(gs[1, 0])
    ax_imp   = fig.add_subplot(gs[1, 1])

    plot_decision_histogram(pipeline, X, ax=ax_hist)
    plot_pca_projection(    pipeline, X, ax=ax_pca)
    plot_nu_sweep(          pipeline, X, ax=ax_nu)
    plot_feature_importance(pipeline, X, feature_names=feature_names, top_n=15, ax=ax_imp)

    fig.suptitle("One-Class SVM diagnostics", fontsize=14, fontweight="bold")

    if out_dir:
        dashboard_path = out_dir / "svm_dashboard.png"
        fig.savefig(dashboard_path, dpi=150, bbox_inches="tight")
        print(f"\n── Dashboard saved → {dashboard_path}")

        # Also save individual plots
        plot_decision_histogram(pipeline, X, save_path=out_dir / "1_decision_histogram.png")
        plot_pca_projection(    pipeline, X, save_path=out_dir / "2_pca_projection.png")
        plot_nu_sweep(          pipeline, X, save_path=out_dir / "3_nu_sweep.png")
        plot_feature_importance(pipeline, X, feature_names=feature_names, top_n=20,
                                save_path=out_dir / "4_feature_importance.png")

    plt.show()
