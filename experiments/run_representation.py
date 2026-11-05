"""Run the representation experiment. Execute from the repo root:
python -m experiments.run_representation --seed 42 --save_dir outputs/

Answers directive Part A on train/val only (the test split is never loaded into any computation here):
intrinsic dimensionality, components for 90/95% variance, reconstruction MSE vs k, and 2D/3D class separation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from src.data_loader import CLASS_NAMES, IMAGE_SHAPE, load_splits, stratified_subsample
from src.evaluate import (
    CLASS_COLORS, DIVERGING, INK, INK_MUTED, INK_SECONDARY, SERIES, SURFACE, apply_style, knn_accuracy,
    plot_class_small_multiples, save_csv, save_figure, save_json,
)
from src.representation import (
    components_for_variance, embed_2d, fit_pca, mle_intrinsic_dimension, project, reconstruct,
    reconstruction_mse,
)
from src.utils import ROOT, set_seed

VARIANCE_THRESHOLDS = (0.90, 0.95, 0.99)
MSE_KS = [1, 2, 3, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500, 600, 784]
SEPARABILITY_KS = [2, 3, 5, 10, 20, 50, 100]
INTERIM_KS = [2, 10, 50, 100, 200]
RECON_SAMPLE_CLASSES = [0, 1, 5, 8, 9]  # T-shirt/top, Trouser, Sandal, Bag, Ankle boot
VIS_SAMPLE_SIZE = 5_000
TSNE_PERPLEXITY = 30.0
SILHOUETTE_SAMPLE_SIZE = 5_000
TSNE_PCA_DIM = 50
MLE_NEIGHBOURS = (10, 20)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=Path, default=ROOT / "outputs",
                        help="Output root; figures go to <save_dir>/figures, metrics to <save_dir>/metrics")
    return parser.parse_args(argv)


def clear_image_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_explained_variance(cumulative: np.ndarray, ratios: np.ndarray, k_for: dict[float, int]) -> plt.Figure:
    ks = np.arange(1, len(ratios) + 1)
    fig, (ax_scree, ax_cum) = plt.subplots(1, 2, figsize=(11, 4))

    ax_scree.plot(ks, ratios, color=SERIES[0], label="Explained variance ratio")
    ax_scree.set_yscale("log")
    ax_scree.set_xlabel("Principal component index")
    ax_scree.set_ylabel("Explained variance ratio (log scale)")
    ax_scree.set_title("Scree plot")
    ax_scree.legend()

    ax_cum.plot(ks, cumulative, color=SERIES[0], label="Cumulative explained variance")
    for (threshold, k), color in zip(k_for.items(), (SERIES[1], SERIES[6], INK_MUTED)):
        ax_cum.axhline(threshold, color=color, linewidth=1, linestyle="--")
        ax_cum.axvline(k, color=color, linewidth=1, linestyle="--", label=f"{threshold:.0%} variance: k = {k}")
    ax_cum.set_xlabel("Number of principal components k")
    ax_cum.set_ylabel("Cumulative explained variance")
    ax_cum.set_ylim(0, 1.02)
    ax_cum.set_title("Cumulative explained variance")
    ax_cum.legend(loc="lower right")

    fig.suptitle("PCA spectrum of Fashion-MNIST (fit on 50,000 training images)", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_reconstruction_mse(mse_df: pd.DataFrame, k_for: dict[float, int]) -> plt.Figure:
    plotted = mse_df[mse_df["k"] < mse_df["k"].max()]
    max_rel_gap = float(((plotted["mse_val"] - plotted["mse_train"]).abs() / plotted["mse_train"]).max())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(plotted["k"], plotted["mse_val"], color=SERIES[1], linewidth=3.5, label="Validation")
    ax.plot(plotted["k"], plotted["mse_train"], color=SERIES[0], marker="o", markersize=4, linewidth=1.5,
            label="Train")
    for (threshold, k), color in zip(k_for.items(), (SERIES[1], SERIES[6], INK_MUTED)):
        ax.axvline(k, color=color, linewidth=1, linestyle="--", label=f"{threshold:.0%} variance (k = {k})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Latent dimension k (log scale)")
    ax.set_ylabel("Per-pixel reconstruction MSE (log scale)")
    ax.set_title("PCA reconstruction error vs latent dimension")
    ax.text(0.02, 0.04, f"Train and validation curves coincide (max relative gap {max_rel_gap:.1%});\n"
            "k = 784 omitted (exact reconstruction, MSE ≈ 0)",
            transform=ax.transAxes, fontsize=8, color=INK_SECONDARY)
    ax.legend(loc="center left", bbox_to_anchor=(0.01, 0.42))
    fig.tight_layout()
    return fig


def plot_reconstructions(
    pca, X: np.ndarray, y: np.ndarray, ks: list[int], cumulative: np.ndarray,
    classes: list[int] | None = None, show_mse: bool = False, title: str | None = None,
) -> plt.Figure:
    classes = list(range(len(CLASS_NAMES))) if classes is None else classes
    rows = [int(np.flatnonzero(y == c)[0]) for c in classes]
    n_cols = len(ks) + 1
    scale = 1.6 if show_mse else 1.3
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(scale * n_cols, (scale + 0.1) * len(rows)))
    for r, idx in enumerate(rows):
        images = [X[idx]] + [reconstruct(pca, X[idx:idx + 1], k)[0] for k in ks]
        for c, image in enumerate(images):
            ax = axes[r, c]
            ax.imshow(np.clip(image, 0, 1).reshape(IMAGE_SHAPE), cmap="gray_r", vmin=0, vmax=1)
            clear_image_axis(ax)
            if r == 0:
                ax.set_title("Original" if c == 0 else f"k = {ks[c - 1]}\n({cumulative[ks[c - 1] - 1]:.0%} var.)",
                             fontsize=8)
            if c == 0:
                ax.set_ylabel(CLASS_NAMES[y[idx]], fontsize=8, rotation=0, ha="right", va="center")
            elif show_mse:
                ax.set_xlabel(f"MSE {np.mean((image - X[idx]) ** 2):.4f}", fontsize=7, color=INK_SECONDARY)
    fig.suptitle(title or "PCA reconstructions of validation images at increasing latent dimension k",
                 x=0.01, ha="left")
    fig.tight_layout()
    return fig


def label_positions(
    ax: plt.Axes, points: np.ndarray, min_dx: float = 0.13, min_dy: float = 0.045
) -> list[tuple[float, float]]:
    """Axes-fraction label positions at `points`, nudged vertically until no two labels overlap."""
    ax.autoscale_view()
    anchors = ax.transAxes.inverted().transform(ax.transData.transform(points))
    placed: list[tuple[float, float]] = []
    for x, y0 in anchors:
        y, step = y0, 0
        while any(abs(x - px) < min_dx and abs(y - py) < min_dy for px, py in placed):
            step += 1
            y = y0 + min_dy * ((step + 1) // 2) * (1 if step % 2 else -1)
        placed.append((float(x), float(y)))
    return placed


def plot_manifold_comparison(
    Z_pca: np.ndarray, Z_tsne: np.ndarray, y: np.ndarray, ratios: np.ndarray, perplexity: float
) -> plt.Figure:
    """PCA-2D and t-SNE-2D side by side, coloured by class. Ten classes exceed the colour-safe palette,
    so every class is also direct-labelled at its median position."""
    fig, (ax_pca, ax_tsne) = plt.subplots(1, 2, figsize=(13, 6.2))
    panels = (
        (ax_pca, Z_pca, f"PCA (linear): PC1–PC2, {ratios[0] + ratios[1]:.1%} of variance",
         f"PC1 ({ratios[0]:.1%} var.)", f"PC2 ({ratios[1]:.1%} var.)"),
        (ax_tsne, Z_tsne, f"t-SNE (non-linear): perplexity {perplexity:g}, on PCA-{TSNE_PCA_DIM}",
         "t-SNE dimension 1", "t-SNE dimension 2"),
    )
    halo = [path_effects.withStroke(linewidth=3, foreground=SURFACE)]
    for ax, Z, title, xlabel, ylabel in panels:
        for c, name in enumerate(CLASS_NAMES):
            mask = y == c
            ax.scatter(Z[mask, 0], Z[mask, 1], s=3, color=CLASS_COLORS[c], alpha=0.7, linewidths=0,
                       label=name, rasterized=True)
        medians = np.array([np.median(Z[y == c], axis=0) for c in range(len(CLASS_NAMES))])
        for name, (lx, ly) in zip(CLASS_NAMES, label_positions(ax, medians)):
            ax.text(lx, ly, name, fontsize=8, fontweight="bold", color=INK, ha="center", va="center",
                    path_effects=halo, transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(False)
    handles, labels = ax_pca.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=5, markerscale=4, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"2D manifold of {len(y):,} stratified training images: PCA vs t-SNE (labels at class medians)",
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def plot_class_means(X: np.ndarray, y: np.ndarray) -> plt.Figure:
    fig, axes = plt.subplots(1, len(CLASS_NAMES), figsize=(13, 1.9))
    for c, ax in enumerate(axes):
        ax.imshow(X[y == c].mean(axis=0).reshape(IMAGE_SHAPE), cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(CLASS_NAMES[c], fontsize=8)
        clear_image_axis(ax)
    fig.suptitle("Mean training image per class (pixel intensity 0–1)", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_components(pca, n: int = 10) -> plt.Figure:
    components = pca.components_[:n]
    limit = np.abs(components).max()
    fig, axes = plt.subplots(2, n // 2, figsize=(12, 5.4))
    for i, ax in enumerate(axes.flat):
        im = ax.imshow(components[i].reshape(IMAGE_SHAPE), cmap=DIVERGING, vmin=-limit, vmax=limit)
        ax.set_title(f"PC{i + 1} ({pca.explained_variance_ratio_[i]:.1%} var.)", fontsize=9)
        clear_image_axis(ax)
    fig.suptitle("First 10 principal components as images", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 0.92, 1))
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("Component loading (blue = negative, red = positive)")
    return fig


def plot_separability(sep_df: pd.DataFrame) -> plt.Figure:
    pca_rows = sep_df[sep_df["space"] == "pca"]
    raw_row = sep_df[sep_df["space"] == "raw"].iloc[0]
    fig, (ax_knn, ax_sil) = plt.subplots(1, 2, figsize=(11, 4))
    for ax, column, label in (
        (ax_knn, "knn_val_accuracy", "5-NN validation accuracy"),
        (ax_sil, "label_silhouette_val", "Silhouette of true labels (validation)"),
    ):
        ax.plot(pca_rows["dim"], pca_rows[column], color=SERIES[0], marker="o", markersize=5, label="PCA-k space")
        ax.axhline(raw_row[column], color=INK_SECONDARY, linewidth=1, linestyle="--",
                   label=f"Raw pixels (784-D): {raw_row[column]:.3f}")
        ax.set_xscale("log")
        ax.set_xlabel("PCA dimension k (log scale)")
        ax.set_ylabel(label)
        ax.legend(loc="lower right")
    ax_knn.set_title("Class separability: nearest-neighbour accuracy")
    ax_sil.set_title("Class separability: label silhouette")
    fig.suptitle("How class separation changes with PCA dimension (fit on train, scored on validation)",
                 x=0.01, ha="left")
    fig.tight_layout()
    return fig


def build_interim_metrics(
    seed: int, train_shape: tuple[int, int], k_for: dict[float, int], cumulative: np.ndarray, mse_df: pd.DataFrame
) -> dict:
    """Interim-submission metrics: 90/95% thresholds, MSE table at INTERIM_KS + k90/k95, and run settings."""
    k95 = k_for[0.95]
    interim_ks = sorted(set(INTERIM_KS) | {k_for[0.90], k95})
    interim_mse = mse_df[mse_df["k"].isin(interim_ks)].reset_index(drop=True)
    return {
        "seed": seed,
        "pca": {"fit_on": "train", "n_train": train_shape[0], "n_features": train_shape[1], "svd_solver": "full"},
        "variance_thresholds": {
            f"{t:.0%}": {"n_components": k_for[t], "cumulative_explained_variance": float(cumulative[k_for[t] - 1])}
            for t in (0.90, 0.95)
        },
        "reconstruction_mse": interim_mse.to_dict(orient="records"),
        "reconstruction_samples": {"classes": [CLASS_NAMES[c] for c in RECON_SAMPLE_CLASSES], "ks": [10, 50, k95],
                                   "split": "val"},
        "manifold_2d": {"n_samples": VIS_SAMPLE_SIZE, "stratified": True, "split": "train",
                        "tsne_perplexity": TSNE_PERPLEXITY, "tsne_input_pca_dim": TSNE_PCA_DIM, "tsne_seed": seed},
    }


def main(seed: int = 42, save_dir: Path = ROOT / "outputs") -> None:
    set_seed(seed)
    apply_style()
    figures_dir, metrics_dir = Path(save_dir) / "figures", Path(save_dir) / "metrics"
    splits = load_splits(seed=seed)
    X_train, y_train, X_val, y_val = splits.X_train, splits.y_train, splits.X_val, splits.y_val

    pca = fit_pca(X_train, seed=seed)
    ratios = pca.explained_variance_ratio_
    cumulative = np.cumsum(ratios)
    k_for = components_for_variance(pca, VARIANCE_THRESHOLDS)
    k90, k95 = k_for[0.90], k_for[0.95]

    mse_ks = sorted(set(MSE_KS) | set(INTERIM_KS) | set(k_for.values()))
    mse_df = pd.DataFrame({
        "k": mse_ks,
        "cumulative_explained_variance": [float(cumulative[k - 1]) for k in mse_ks],
        "mse_train": reconstruction_mse(pca, X_train, mse_ks),
        "mse_val": reconstruction_mse(pca, X_val, mse_ks),
    })
    save_csv(mse_df, "reconstruction_mse.csv", metrics_dir)

    mle_ids = {n: mle_intrinsic_dimension(X_train, n_neighbors=n, seed=seed) for n in MLE_NEIGHBOURS}

    sil_idx = stratified_subsample(y_val, SILHOUETTE_SAMPLE_SIZE, seed=seed)
    sep_rows = []
    for k in sorted(set(SEPARABILITY_KS) | {k90, k95}):
        Z_train, Z_val = project(pca, X_train, k), project(pca, X_val, k)
        sep_rows.append({
            "space": "pca", "dim": k, "cumulative_explained_variance": float(cumulative[k - 1]),
            "knn_val_accuracy": knn_accuracy(Z_train, y_train, Z_val, y_val),
            "label_silhouette_val": float(silhouette_score(Z_val[sil_idx], y_val[sil_idx])),
        })
    sep_rows.append({
        "space": "raw", "dim": X_train.shape[1], "cumulative_explained_variance": 1.0,
        "knn_val_accuracy": knn_accuracy(X_train, y_train, X_val, y_val),
        "label_silhouette_val": float(silhouette_score(X_val[sil_idx], y_val[sil_idx])),
    })
    sep_df = pd.DataFrame(sep_rows)
    save_csv(sep_df, "separability_vs_dim.csv", metrics_dir)

    vis_idx, Z_pca_vis, Z_tsne = embed_2d(
        pca, X_train, y_train, n_samples=VIS_SAMPLE_SIZE, perplexity=TSNE_PERPLEXITY,
        tsne_pca_dim=TSNE_PCA_DIM, seed=seed,
    )

    save_figure(plot_explained_variance(cumulative, ratios, k_for), "pca_explained_variance.png", figures_dir)
    save_figure(plot_reconstruction_mse(mse_df, k_for), "reconstruction_mse_vs_k.png", figures_dir)
    save_figure(plot_reconstructions(pca, X_val, y_val, [2, 10, 50, k90, k95, 300], cumulative),
                "pca_reconstructions.png", figures_dir)
    save_figure(plot_reconstructions(
        pca, X_val, y_val, [10, 50, k95], cumulative, classes=RECON_SAMPLE_CLASSES, show_mse=True,
        title="Original vs PCA-reconstructed validation images (k = 10, 50, k₉₅)",
    ), "reconstruction_samples.png", figures_dir)
    save_figure(plot_manifold_comparison(Z_pca_vis, Z_tsne, y_train[vis_idx], ratios, TSNE_PERPLEXITY),
                "manifold_2d_pca_vs_tsne.png", figures_dir)
    save_figure(plot_class_means(X_train, y_train), "class_means.png", figures_dir)
    save_figure(plot_components(pca), "pca_components.png", figures_dir)
    save_figure(plot_class_small_multiples(
        Z_pca_vis, y_train[vis_idx], CLASS_NAMES,
        f"PCA projection onto PC1–PC2 ({cumulative[1]:.1%} of variance), {VIS_SAMPLE_SIZE:,} training images",
        f"PC1 ({ratios[0]:.1%} var.)", f"PC2 ({ratios[1]:.1%} var.)",
    ), "pca_2d_by_class.png", figures_dir)
    save_figure(plot_class_small_multiples(
        Z_tsne, y_train[vis_idx], CLASS_NAMES,
        f"t-SNE of the same {VIS_SAMPLE_SIZE:,} training images (on PCA-{TSNE_PCA_DIM}, perplexity {TSNE_PERPLEXITY:g})",
        "t-SNE dimension 1", "t-SNE dimension 2",
    ), "tsne_2d_by_class.png", figures_dir)
    save_figure(plot_separability(sep_df), "separability_vs_dim.png", figures_dir)

    sep_by_dim = sep_df[sep_df["space"] == "pca"].set_index("dim")
    mse_by_k = mse_df.set_index("k")
    metrics = {
        "seed": seed,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "pca": {"svd_solver": "full", "preprocessing": "centred on train mean, pixels scaled by 1/255"},
        "components_for_variance": {f"{t:.0%}": k for t, k in k_for.items()},
        "explained_variance": {f"k={k}": float(cumulative[k - 1]) for k in (1, 2, 3, 10, 50, 100)},
        "reconstruction_mse": {
            f"k={k}": {"train": float(mse_by_k.loc[k, "mse_train"]), "val": float(mse_by_k.loc[k, "mse_val"])}
            for k in (2, 10, 50, k90, k95)
        },
        "intrinsic_dimension_mle": {f"n_neighbors={n}": d for n, d in mle_ids.items()},
        "separability_val": {
            "knn5_accuracy": {f"pca_{k}": float(sep_by_dim.loc[k, "knn_val_accuracy"]) for k in (2, 3, 10, 50)},
            "label_silhouette": {f"pca_{k}": float(sep_by_dim.loc[k, "label_silhouette_val"]) for k in (2, 3, 10, 50)},
            "raw_knn5_accuracy": float(sep_df.iloc[-1]["knn_val_accuracy"]),
            "raw_label_silhouette": float(sep_df.iloc[-1]["label_silhouette_val"]),
        },
        "visualisation": {"sample_size": VIS_SAMPLE_SIZE, "tsne_input_pca_dim": TSNE_PCA_DIM,
                          "tsne_perplexity": TSNE_PERPLEXITY},
    }
    save_json(metrics, "representation_metrics.json", metrics_dir)

    interim = build_interim_metrics(seed, X_train.shape, k_for, cumulative, mse_df)
    interim_mse = pd.DataFrame(interim["reconstruction_mse"])
    save_json(interim, "representation_interim.json", metrics_dir)

    print(f"Components: 90% variance -> k = {k90}, 95% variance -> k = {k95}, 99% -> k = {k_for[0.99]}")
    print("Reconstruction MSE (per pixel):")
    print(interim_mse.to_string(index=False))
    print(f"MLE intrinsic dimension: {mle_ids}")
    print(sep_df.to_string(index=False))


if __name__ == "__main__":
    args = parse_args()
    main(args.seed, args.save_dir)
