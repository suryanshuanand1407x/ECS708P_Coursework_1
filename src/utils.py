"""Shared helpers: seeding and output paths."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "outputs" / "figures"
METRICS_DIR = ROOT / "outputs" / "metrics"


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch RNGs and request deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
