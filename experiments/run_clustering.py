"""Run the clustering experiment. Execute from the repo root: python -m experiments.run_clustering --seed 42"""

from __future__ import annotations

import argparse

from src.utils import set_seed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(seed: int = 42) -> None:
    set_seed(seed)
    raise NotImplementedError


if __name__ == "__main__":
    main(parse_args().seed)
