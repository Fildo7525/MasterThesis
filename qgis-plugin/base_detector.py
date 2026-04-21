"""
base_detector.py
----------------
Abstract base class that every one-class detector must implement.

Both IsolationForest and GMM detectors inherit from this, so NgrviApproach
only needs to talk to the base interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class BaseDetector(ABC):
    """
    Minimal interface expected by NgrviApproach.process_window().

    Subclasses must implement:
        fit(X)         — train on inlier data
        predict(X)     — return +1 (inlier) / -1 (outlier) per row
        score(X)       — return a scalar anomaly score per row
                         (higher = more normal, matching sklearn OCSVM convention)
    """

    # ------------------------------------------------------------------
    # Abstract API
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, X: np.ndarray) -> "BaseDetector":
        """Fit the model on the inlier feature matrix X (n_samples, n_features)."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return an integer array of shape (n_samples,).
        Convention (same as sklearn OneClassSVM):
            +1  →  inlier  (belongs to the group)
            -1  →  outlier
        """

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Return a float array of shape (n_samples,).
        Higher values mean more likely to be an inlier, so that the
        existing debug print  score={score:+.4f}  stays meaningful.
        """

    # ------------------------------------------------------------------
    # Convenience helpers (shared by all subclasses)
    # ------------------------------------------------------------------

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Alias for score() so that NgrviApproach can call either
        pipeline.decision_function(X) or detector.decision_function(X)
        without caring which concrete class it holds.
        """
        return self.score(X)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_fitted(attr: Any, name: str = "model") -> None:
        if attr is None:
            raise RuntimeError(
                f"{name} is not fitted yet. Call fit() before predict() / score()."
            )
