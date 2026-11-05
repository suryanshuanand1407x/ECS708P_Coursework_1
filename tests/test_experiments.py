"""Rigor, shape, and leakage assertions."""

import gzip
import importlib

import numpy as np
import pytest

from src.clustering import purity_score
from src.data_loader import FILES, RAW_DIR, load_splits, make_split_indices, read_idx

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
