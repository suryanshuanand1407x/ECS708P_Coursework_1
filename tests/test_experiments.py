"""Rigor, shape, and leakage assertions."""

import gzip
import importlib

import numpy as np
import pytest

from src.clustering import purity_score
from src.data_loader import FILES, RAW_DIR, load_splits, make_split_indices, read_idx, stratified_subsample
from src.representation import (
    components_for_variance, embed_2d, fit_pca, mle_intrinsic_dimension, project, reconstruct, reconstruction_mse,
    tsne_embed,
)

MODULES = [
    "src.data_loader",
    "src.representation",
    "src.clustering",
    "src.classification",
    "src.evaluate",
    "src.utils",
]

RAW_DATA_PRESENT = all((RAW_DIR / filename).exists() for filename, _ in FILES.values())


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module: str) -> None:
    importlib.import_module(module)


def test_read_idx_roundtrip(tmp_path) -> None:
    images = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    header = b"\x00\x00\x08\x03" + b"".join(d.to_bytes(4, "big") for d in images.shape)
    path = tmp_path / "images.gz"
    with gzip.open(path, "wb") as f:
        f.write(header + images.tobytes())
    np.testing.assert_array_equal(read_idx(path), images)


def test_split_is_disjoint_stratified_and_deterministic() -> None:
    y = np.repeat(np.arange(10), 600)
    train_a, val_a = make_split_indices(y, val_size=1000, seed=42)
    train_b, val_b = make_split_indices(y, val_size=1000, seed=42)

    np.testing.assert_array_equal(train_a, train_b)
    np.testing.assert_array_equal(val_a, val_b)
    assert np.intersect1d(train_a, val_a).size == 0
    assert len(train_a) + len(val_a) == len(y)
    assert (np.bincount(y[val_a]) == 100).all()


def test_split_changes_with_seed() -> None:
    y = np.repeat(np.arange(10), 600)
    _, val_42 = make_split_indices(y, val_size=1000, seed=42)
    _, val_0 = make_split_indices(y, val_size=1000, seed=0)
    assert not np.array_equal(val_42, val_0)


def test_purity_score() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1])
    assert purity_score(y_true, np.array([0, 0, 0, 1, 1, 1])) == 1.0
    assert purity_score(y_true, np.array([0, 0, 1, 1, 1, 1])) == pytest.approx(5 / 6)


@pytest.mark.skipif(not RAW_DATA_PRESENT, reason="Fashion-MNIST not downloaded")
def test_load_splits_shapes_and_ranges(tmp_path) -> None:
    splits = load_splits(seed=42, split_path=tmp_path / "split.npz")

    assert splits.X_train.shape == (50_000, 784)
    assert splits.X_val.shape == (10_000, 784)
    assert splits.X_test.shape == (10_000, 784)
    for X in (splits.X_train, splits.X_val, splits.X_test):
        assert X.dtype == np.float32 and X.min() >= 0.0 and X.max() <= 1.0
    for y in (splits.y_train, splits.y_val, splits.y_test):
        assert set(np.unique(y)) == set(range(10))
    assert (np.bincount(splits.y_val) == 1000).all()
    assert (np.bincount(splits.y_test) == 1000).all()


def _low_rank_data(n: int = 600, dim: int = 40, rank: int = 5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, rank)) @ rng.normal(size=(rank, dim)) + 3.0


def test_stratified_subsample_balanced_and_deterministic() -> None:
    y = np.repeat(np.arange(10), 500)
    idx = stratified_subsample(y, 1000, seed=42)
    np.testing.assert_array_equal(idx, stratified_subsample(y, 1000, seed=42))
    assert (np.bincount(y[idx]) == 100).all()
    assert len(np.unique(idx)) == 1000


def test_pca_statistics_come_from_train_only() -> None:
    X = _low_rank_data()
    X_train, X_val = X[:400], X[400:] + 100.0
    pca = fit_pca(X_train, seed=42)
    np.testing.assert_allclose(pca.mean_, X_train.mean(axis=0))
    assert not np.allclose(pca.mean_, np.vstack([X_train, X_val]).mean(axis=0))


def test_components_for_variance_on_low_rank_data() -> None:
    pca = fit_pca(_low_rank_data(rank=5), seed=42)
    k_for = components_for_variance(pca, (0.90, 0.999999))
    assert k_for[0.999999] == 5
    assert 1 <= k_for[0.90] <= 5


def test_reconstruction_error_decreases_and_vanishes_at_full_rank() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 20))
    pca = fit_pca(X, seed=42)
    mse = reconstruction_mse(pca, X, [1, 5, 10, 20], batch_size=128)
    assert all(a > b for a, b in zip(mse, mse[1:]))
    assert mse[-1] < 1e-20
    np.testing.assert_allclose(reconstruct(pca, X, 20), X, atol=1e-10)


def test_mle_intrinsic_dimension_recovers_linear_subspace() -> None:
    X = _low_rank_data(n=2000, dim=40, rank=5)
    assert 4.0 < mle_intrinsic_dimension(X, n_neighbors=20, sample_size=2000, seed=42) < 6.0


def test_tsne_is_deterministic_for_fixed_seed() -> None:
    X = _low_rank_data(n=150, dim=10, rank=3)
    np.testing.assert_array_equal(tsne_embed(X, perplexity=10, seed=42), tsne_embed(X, perplexity=10, seed=42))


def _labelled_low_rank_data(n_per_class: int = 40, n_classes: int = 10, dim: int = 30, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(n_classes), n_per_class)
    centres = rng.normal(scale=5.0, size=(n_classes, dim))
    return centres[y] + rng.normal(size=(len(y), dim)), y


@pytest.mark.parametrize("k", [1, 2, 10, 30])
def test_projection_and_reconstruction_shapes(k: int) -> None:
    X, _ = _labelled_low_rank_data()
    X_train, X_val = X[:300], X[300:]
    pca = fit_pca(X_train, seed=42)
    assert project(pca, X_val, k).shape == (len(X_val), k)
    assert reconstruct(pca, X_val, k).shape == X_val.shape
    assert len(reconstruction_mse(pca, X_val, [k])) == 1


def test_embed_2d_shapes_and_stratified_subsample() -> None:
    X, y = _labelled_low_rank_data()
    pca = fit_pca(X, seed=42)
    idx, Z_pca, Z_tsne = embed_2d(pca, X, y, n_samples=200, perplexity=10, tsne_pca_dim=5, seed=42)
    assert idx.shape == (200,)
    assert Z_pca.shape == (200, 2) and Z_tsne.shape == (200, 2)
    assert (np.bincount(y[idx]) == 20).all()
    np.testing.assert_allclose(Z_pca, project(pca, X[idx], 2))


@pytest.mark.skipif(not RAW_DATA_PRESENT, reason="Fashion-MNIST not downloaded")
def test_pca_on_fashion_mnist_shapes(tmp_path) -> None:
    splits = load_splits(seed=42, split_path=tmp_path / "split.npz")
    pca = fit_pca(splits.X_train[:5_000], n_components=200, seed=42)
    assert pca.components_.shape == (200, 784)
    k_for = components_for_variance(pca, (0.90,))
    assert project(pca, splits.X_val, k_for[0.90]).shape == (10_000, k_for[0.90])
    assert reconstruct(pca, splits.X_val[:100], 50).shape == (100, 784)
