"""Fashion-MNIST loading and the fixed train/val/test split.

The official 60k training set is split (stratified) into train and val; the
official 10k test set is kept untouched as the out-of-sample holdout.
Pixels are scaled by the constant 1/255, which is not fitted to any data.
"""

from __future__ import annotations

import gzip
import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "fashion-mnist"
SPLIT_PATH = ROOT / "data" / "processed" / "split_indices.npz"

BASE_URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
FILES = {
    "train_images": ("train-images-idx3-ubyte.gz", "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
    "train_labels": ("train-labels-idx1-ubyte.gz", "25c81989df183df01b3e8a0aad5dffbe"),
    "test_images": ("t10k-images-idx3-ubyte.gz", "bef4ecab320f06d8554ea6380940ec79"),
    "test_labels": ("t10k-labels-idx1-ubyte.gz", "bb300cfdad3c16e7a12a480ee83cd310"),
}

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
IMAGE_SHAPE = (28, 28)


@dataclass(frozen=True)
class DataSplits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def download_fashion_mnist(raw_dir: Path = RAW_DIR) -> None:
    """Download any missing files and verify every file's MD5 checksum."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for filename, md5 in FILES.values():
        path = raw_dir / filename
        if not path.exists():
            urllib.request.urlretrieve(BASE_URL + filename, path)
        if _md5(path) != md5:
            raise ValueError(f"Checksum mismatch for {path}; delete it and re-run.")


def read_idx(path: Path) -> np.ndarray:
    """Parse a gzipped IDX file of unsigned bytes into an array of its stored shape."""
    with gzip.open(path, "rb") as f:
        data = f.read()
    zero, dtype_code, ndim = data[0:2], data[2], data[3]
    if zero != b"\x00\x00" or dtype_code != 0x08:
        raise ValueError(f"{path} is not an unsigned-byte IDX file.")
    shape = tuple(int.from_bytes(data[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim))
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


def load_raw(raw_dir: Path = RAW_DIR) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the official (X_train, y_train, X_test, y_test) with images flattened to 784 uint8 pixels."""
    download_fashion_mnist(raw_dir)
    arrays = {key: read_idx(raw_dir / filename) for key, (filename, _) in FILES.items()}
    X_train = arrays["train_images"].reshape(len(arrays["train_images"]), -1)
    X_test = arrays["test_images"].reshape(len(arrays["test_images"]), -1)
    return X_train, arrays["train_labels"], X_test, arrays["test_labels"]


def make_split_indices(
    y: np.ndarray, val_size: int = 10_000, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified train/val index split of the official training set."""
    train_idx, val_idx = train_test_split(
        np.arange(len(y)), test_size=val_size, stratify=y, random_state=seed
    )
    return np.sort(train_idx), np.sort(val_idx)


def stratified_subsample(y: np.ndarray, n: int, seed: int = 42) -> np.ndarray:
    """Sorted indices of a class-balanced subsample of size n."""
    if n >= len(y):
        return np.arange(len(y))
    idx, _ = train_test_split(np.arange(len(y)), train_size=n, stratify=y, random_state=seed)
    return np.sort(idx)


def load_splits(
    val_size: int = 10_000,
    seed: int = 42,
    raw_dir: Path = RAW_DIR,
    split_path: Path = SPLIT_PATH,
) -> DataSplits:
    """Load train/val/test as float32 in [0, 1], reusing the saved split indices when they match."""
    X_full, y_full, X_test, y_test = load_raw(raw_dir)

    cached = np.load(split_path) if split_path.exists() else None
    if cached is not None and int(cached["seed"]) == seed and int(cached["val_size"]) == val_size:
        train_idx, val_idx = cached["train_idx"], cached["val_idx"]
    else:
        train_idx, val_idx = make_split_indices(y_full, val_size=val_size, seed=seed)
        split_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(split_path, train_idx=train_idx, val_idx=val_idx, seed=seed, val_size=val_size)

    def scale(X: np.ndarray) -> np.ndarray:
        return X.astype(np.float32) / 255.0

    return DataSplits(
        X_train=scale(X_full[train_idx]), y_train=y_full[train_idx].astype(np.int64),
        X_val=scale(X_full[val_idx]), y_val=y_full[val_idx].astype(np.int64),
        X_test=scale(X_test), y_test=y_test.astype(np.int64),
    )
