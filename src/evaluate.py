"""Evaluation metrics, figure styling and output writers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.neighbors import KNeighborsClassifier

from src.utils import FIGURES_DIR, METRICS_DIR

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
CONTEXT_GRAY = "#c3c2b7"
# Ten classes exceed the 8 validated categorical slots, so colour alone cannot carry identity:
# any figure using CLASS_COLORS must also direct-label each class.
CLASS_COLORS = SERIES + ["#52514e", "#9c6b30"]
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#184f95", "#f0efec", "#e34948"])
SEQUENTIAL = LinearSegmentedColormap.from_list("blue_seq", ["#fcfcfb", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"])

DPI = 300


def apply_style() -> None:
    """Set the shared matplotlib style used by every report figure."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2,
        "figure.titlesize": 13,
    })


def save_figure(fig: plt.Figure, name: str, figures_dir: Path = FIGURES_DIR) -> Path:
    """Save a figure at 300 DPI to outputs/figures/ and close it."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def save_json(obj: dict, name: str, metrics_dir: Path = METRICS_DIR) -> Path:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    path = metrics_dir / name
    path.write_text(json.dumps(obj, indent=2, default=_to_builtin) + "\n")
    return path


def save_csv(df: pd.DataFrame, name: str, metrics_dir: Path = METRICS_DIR) -> Path:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    path = metrics_dir / name
    df.to_csv(path, index=False)
    return path


def _to_builtin(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialise {type(value)}")


def plot_class_small_multiples(
    Z: np.ndarray, y: np.ndarray, class_names: list[str], title: str, xlabel: str, ylabel: str
) -> plt.Figure:
    """One panel per class: that class highlighted over all other points in gray, on shared axes."""
    n_cols = 5
    n_rows = int(np.ceil(len(class_names) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), sharex=True, sharey=True)
    for c, ax in enumerate(axes.flat):
        if c >= len(class_names):
            ax.set_visible(False)
            continue
        mask = y == c
        ax.scatter(Z[~mask, 0], Z[~mask, 1], s=1, c=CONTEXT_GRAY, alpha=0.35, linewidths=0, rasterized=True)
        ax.scatter(Z[mask, 0], Z[mask, 1], s=2, c=SERIES[0], alpha=0.8, linewidths=0, rasterized=True)
        ax.set_title(class_names[c])
        ax.grid(False)
    for ax in axes[-1]:
        ax.set_xlabel(xlabel)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=SERIES[0], label="Highlighted class"),
        plt.Line2D([], [], marker="o", linestyle="", color=CONTEXT_GRAY, label="All other classes"),
    ]
    fig.legend(handles=handles, loc="upper right", ncols=2, bbox_to_anchor=(1.0, 1.0))
    fig.suptitle(title, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def knn_accuracy(
    Z_train: np.ndarray, y_train: np.ndarray, Z_eval: np.ndarray, y_eval: np.ndarray, n_neighbors: int = 5
) -> float:
    """Accuracy of a k-NN classifier fit on train embeddings: a model-light probe of class separability."""
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1).fit(Z_train, y_train)
    return float(knn.score(Z_eval, y_eval))
