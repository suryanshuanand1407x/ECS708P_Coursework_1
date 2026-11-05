"""Latent space and dimensionality reduction: PCA, intrinsic dimension, t-SNE."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

from src.data_loader import stratified_subsample


def fit_pca(X_train: np.ndarray, n_components: int | None = None, seed: int = 42) -> PCA:
    """Fit PCA on the training split only. Pixels share units, so data is centred but not standardised."""
    return PCA(n_components=n_components, svd_solver="full", random_state=seed).fit(X_train)


def components_for_variance(pca: PCA, thresholds: tuple[float, ...] = (0.90, 0.95, 0.99)) -> dict[float, int]:
    """Smallest number of components whose cumulative explained variance reaches each threshold."""
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    return {t: int(np.searchsorted(cumulative, t) + 1) for t in thresholds}


def project(pca: PCA, X: np.ndarray, k: int) -> np.ndarray:
    """Coordinates of X on the first k principal components."""
    return (X - pca.mean_) @ pca.components_[:k].T


def reconstruct(pca: PCA, X: np.ndarray, k: int) -> np.ndarray:
    """Reconstruct X from its projection on the first k principal components."""
    return project(pca, X, k) @ pca.components_[:k] + pca.mean_


def reconstruction_mse(pca: PCA, X: np.ndarray, ks: list[int], batch_size: int = 10_000) -> list[float]:
    """Mean squared per-pixel reconstruction error of X for each latent dimension k."""
    totals = np.zeros(len(ks))
    for start in range(0, len(X), batch_size):
        batch = X[start:start + batch_size]
        for i, k in enumerate(ks):
            totals[i] += np.sum((batch - reconstruct(pca, batch, k)) ** 2)
    return (totals / X.size).tolist()


def mle_intrinsic_dimension(
    X: np.ndarray, n_neighbors: int = 20, sample_size: int = 5_000, seed: int = 42
) -> float:
    """Levina-Bickel MLE of intrinsic dimension, pooled with the MacKay-Ghahramani correction.

    Estimated on a seeded subsample because neighbour search is quadratic in sample size.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(sample_size, len(X)), replace=False)
    distances, _ = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X[idx]).kneighbors(X[idx])
    distances = distances[:, 1:]
    distances = distances[(distances > 0).all(axis=1)]
    log_ratios = np.log(distances[:, -1:] / distances[:, :-1])
    inverse_estimates = log_ratios.sum(axis=1) / (n_neighbors - 1)
    return float(1.0 / inverse_estimates.mean())


def tsne_embed(X: np.ndarray, perplexity: float = 30.0, seed: int = 42) -> np.ndarray:
    """2D t-SNE embedding. Transductive: for visualisation only, never as model features."""
    return TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=seed).fit_transform(X)


def embed_2d(
    pca: PCA,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_samples: int = 5_000,
    perplexity: float = 30.0,
    tsne_pca_dim: int = 50,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2D PCA and 2D t-SNE embeddings of the same stratified training subsample.

    t-SNE runs on the first `tsne_pca_dim` principal components to suppress pixel noise.
    Returns (subsample indices, PCA coordinates, t-SNE coordinates).
    """
    idx = stratified_subsample(y_train, n_samples, seed=seed)
    Z_pca = project(pca, X_train[idx], 2)
    Z_tsne = tsne_embed(project(pca, X_train[idx], tsne_pca_dim), perplexity=perplexity, seed=seed)
    return idx, Z_pca, Z_tsne
