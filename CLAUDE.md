# Understanding Visual Data — Coursework Repository

Modular Python ML pipeline: representation learning → clustering → classification → evaluation.
The rules below are **mandatory**. Code that violates them must not be committed.

## Layout

```
coursework_execution_directive.md   Execution guide (phases, questions, definition of done)
ECS708P_interim_representation.ipynb  Interim submission notebook; thin layer over src/ + experiments/ plot functions
docs/                 Coursework brief and notes; REQUIREMENTS_MAP.md maps requirements → experiments → outputs
data/raw/fashion-mnist/  Original Fashion-MNIST .gz files (immutable)
data/processed/       Derived data, incl. split_indices.npz
src/
  data_loader.py      Fashion-MNIST loading, fixed train/val/test split (load_splits)
  representation.py   Feature extraction / dimensionality reduction
  clustering.py       Unsupervised methods and validation metrics (purity_score)
  classification.py   Supervised models
  evaluate.py         Metrics, figure and metric writers
  utils.py            set_seed, FIGURES_DIR, METRICS_DIR
experiments/          Thin CLI runners (all accept --seed); orchestrate src/, contain no core logic
outputs/figures/      All plots (300 DPI)
outputs/metrics/      All results (*.json / *.csv)
tests/test_experiments.py   Rigor, shape, and leakage assertions
```

Use the project virtualenv (pinned versions) and run everything from the repo root:

```bash
source .venv/bin/activate
python -m experiments.run_representation --seed 42
python -m experiments.run_clustering --seed 42
python -m experiments.run_classification --seed 42
pytest
```

## Data

Fashion-MNIST: 28×28 grayscale, 10 balanced classes. Always load through `src.data_loader.load_splits(seed=seed)`,
which returns train 50,000 / val 10,000 (stratified from the official training set) and the official test 10,000,
as float32 in [0, 1]. Do not create other splits. The test split is evaluated **once**, only by the final step of
`run_classification` (`--final`); every other analysis uses train and val.

## 1. Environment

- Python **3.10+**. Use modern syntax (`X | None`, `match`, built-in generics like `list[int]`).
- Allowed libraries: **PyTorch, scikit-learn, numpy, pandas, matplotlib, seaborn, scipy** (plus torchvision/Pillow for image I/O and pytest for tests).
- Do not add a dependency without adding a pinned entry to `requirements.txt`.
- Type-hint all public functions.

## 2. Deterministic execution (`seed=42`)

Every function that performs, or calls something that performs, a stochastic operation
(splitting, shuffling, sampling, augmentation, weight init, k-means init, t-SNE, dropout, DataLoader shuffling, etc.)
**must** accept a `seed` parameter defaulting to `42` and **must propagate it** to every stochastic callee.

```python
def split_data(X, y, test_size: float = 0.2, seed: int = 42):
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

def run_kmeans(X, n_clusters: int, seed: int = 42):
    return KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(X)
```

Rules:
- Never rely on global/implicit randomness: no bare `np.random.*`, `random.*`, or estimators without `random_state`.
  Use `np.random.default_rng(seed)`, `random_state=seed`, `torch.Generator().manual_seed(seed)`.
- Never hard-code a different seed inside a function body; take it from the parameter.
- Each experiment script accepts `--seed` (default 42), calls `src.utils.set_seed(seed)` once at entry,
  and passes `seed` down explicitly.

- PyTorch `DataLoader`s that shuffle must receive `generator=torch.Generator().manual_seed(seed)`
  and, if `num_workers > 0`, a seeded `worker_init_fn`.

## 3. Data leakage — fit on train ONLY

All scalers, encoders, vectorizers, imputers, and dimensionality reduction estimators
(`StandardScaler`, `MinMaxScaler`, `PCA`, `TruncatedSVD`, `LDA`, etc.) **must be fit only on the training split**
and applied to validation/test splits with `.transform()`.

```python
scaler = StandardScaler().fit(X_train)
X_train_s, X_val_s, X_test_s = (scaler.transform(X) for X in (X_train, X_val, X_test))

pca = PCA(n_components=50, random_state=seed).fit(X_train_s)
Z_train, Z_val, Z_test = (pca.transform(X) for X in (X_train_s, X_val_s, X_test_s))
```

Forbidden:
- `fit` / `fit_transform` on the full dataset, on validation, or on test data.
- Fitting before splitting. **Split first, then preprocess.**
- Computing normalisation statistics (e.g. image mean/std) from anything other than the training split.
- Hyperparameter selection on the test split — use validation or cross-validation on train.

Preferred: wrap preprocessing + model in an `sklearn.pipeline.Pipeline` so cross-validation refits per fold.
Transductive methods without a `transform` (e.g. t-SNE) may be used for **visualisation only**, never as features
for a model evaluated on held-out data.

## 4. Figures

- Save every plot to `outputs/figures/` at **300 DPI**: `fig.savefig(path, dpi=300, bbox_inches="tight")`.
- Every figure must have a **descriptive title**, **labeled axes** (with units where applicable), and a **legend**
  whenever more than one series/class/cluster is shown.
- Image grids (samples, reconstructions, components) replace axes with row/column labels; signed images get a colorbar.
- Use the shared style and palette in `src/evaluate.py` (`apply_style`, `SERIES`, `save_figure`). More than 3 classes in
  one scatter → small multiples (`plot_class_small_multiples`), not 10 colours.
- Use descriptive filenames, e.g. `pca_explained_variance.png`, `kmeans_k8_clusters_pca2d.png`.
- Close figures after saving (`plt.close(fig)`); do not call `plt.show()` in pipeline code.

## 5. Metrics

- Log all results to `outputs/metrics/` as `*.json` (single runs / dicts) or `*.csv` (tables, sweeps).
- Include the `seed` and key hyperparameters alongside each result so runs are reproducible.
- Do not report numbers that are only printed to stdout.

## 6. Code conventions

- Core logic lives in `src/`; `experiments/` only wires it together.
- Never write to `data/raw/`.
- Build paths with `pathlib.Path`, relative to the repo root; create output dirs with `mkdir(parents=True, exist_ok=True)`.
- Add or update tests in `tests/` for new pipeline functionality, including leakage and determinism checks
  (same seed ⇒ identical outputs).
