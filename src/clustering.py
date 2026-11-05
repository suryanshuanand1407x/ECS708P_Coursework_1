"""Clustering algorithms and validation metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics.cluster import contingency_matrix


def purity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of samples assigned to the majority true class of their cluster."""
    cm = contingency_matrix(y_true, y_pred)
    return float(cm.max(axis=0).sum() / cm.sum())
