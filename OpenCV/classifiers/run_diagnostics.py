import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_feature_matrix(pretrainer, out_path: Path, feature_names=None, max_features=5):
    """
    Pairplot-style feature matrix:
        - Diagonal: histograms
        - Off-diagonal: scatter plots
        - Inliers: blue
        - Outliers: yellow
    """

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    X = getattr(pretrainer, "last_X", None)
    if X is None:
        raise ValueError("pretrainer.last_X is None")

    model = getattr(pretrainer, "pipeline", None)
    if model is None:
        model = getattr(pretrainer, "detector", None)

    if model is None:
        raise ValueError("No model found in pretrainer")

    # Predictions
    preds = model.predict(X)
    inliers = preds == 1
    outliers = preds == -1

    # Limit features (matrix grows fast)
    n_features = min(max_features, X.shape[1])
    X = X[:, :n_features]

    if feature_names is not None:
        feature_names = feature_names[:n_features]

    fig, axes = plt.subplots(n_features, n_features, figsize=(3*n_features, 3*n_features))

    for i in range(n_features):
        for j in range(n_features):
            ax = axes[i, j]

            if i == j:
                # --- Diagonal: histogram ---
                ax.hist(X[inliers, i], bins=40, alpha=0.7, label="Inliers")
                if outliers.sum() > 0:
                    ax.hist(X[outliers, i], bins=40, alpha=0.7, label="Outliers")

            else:
                # --- Off-diagonal: scatter ---
                ax.scatter(X[inliers, j], X[inliers, i], s=5, label="Inliers")
                if outliers.sum() > 0:
                    ax.scatter(X[outliers, j], X[outliers, i], s=5, label="Outliers")

            # Labels
            if i == n_features - 1:
                ax.set_xlabel(feature_names[j] if feature_names else f"F{j}")
            if j == 0:
                ax.set_ylabel(feature_names[i] if feature_names else f"F{i}")

            ax.tick_params(axis='both', which='both', labelsize=6)

    # Single legend (top-right plot)
    handles, labels = axes[0, -1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    print(f"Feature matrix saved → {out_path}")
