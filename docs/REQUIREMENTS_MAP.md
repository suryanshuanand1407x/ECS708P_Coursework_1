# Requirements Map

> **Status:** draft. The official coursework brief/rubric is **not yet in `docs/`**, so requirements below are taken from
> `coursework_execution_directive.md` only. Re-check every row against the brief once it is added and record any
> mandatory items the directive does not cover.

## Dataset: Fashion-MNIST

| Property | Value |
|---|---|
| Source | https://github.com/zalandoresearch/fashion-mnist (files MD5-verified by `src/data_loader.py`) |
| Images | 28×28 grayscale, flattened to 784 features, scaled by constant 1/255 (not fitted) |
| Classes | 10, balanced: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot |
| Split | official train (60k) → stratified **train 50,000 / val 10,000** (`seed=42`); official **test 10,000** untouched |
| Split file | `data/processed/split_indices.npz` (shared by every phase) |

Leakage protocol: every scaler, PCA and model is fitted on **train** only. Val is used for model selection;
test is evaluated **once**, for the final selected models, via `run_classification --final`.

## Compute budget notes

- RBF-SVM on 50k×784 is slow: tune on a seeded stratified subset and/or PCA features, then refit the chosen config.
- t-SNE: seeded stratified subsample (~5–10k train points), visualisation only.
- Silhouette is O(n²): compute on a seeded subsample (`sample_size`, `random_state=seed`).
- GMM with full covariance on raw 784-D is ill-conditioned: use GMM on PCA spaces (raw space: K-Means only, or diagonal covariance).

## A. Representation & latent structure — `experiments/run_representation.py`

| Requirement | Method | Output |
|---|---|---|
| Intrinsic dimensionality | PCA spectrum, components at 90/95/99% variance, MSE elbow | `representation_metrics.json` |
| Components for 90% and 95% variance | Cumulative explained variance of PCA fit on train | `pca_explained_variance.png`, `representation_metrics.json` |
| Reconstruction MSE vs k | PCA inverse transform, k ∈ {2,5,10,20,50,100,200,300,784}, train and val | `reconstruction_mse_vs_k.png`, `reconstruction_mse.csv` |
| 2D/3D class separation | PCA 2D/3D, t-SNE 2D (subsample), coloured by class | `pca_2d.png`, `pca_3d.png`, `tsne_2d.png` |
| Supporting visuals | Mean image per class, top principal components as images, reconstructions at several k | `class_means.png`, `pca_components.png`, `pca_reconstructions.png` |
| Optional (if brief requires) | Small PyTorch autoencoder, MSE vs latent size compared with PCA | `autoencoder_mse.csv` |

## B. Clustering — `experiments/run_clustering.py`

| Requirement | Method | Output |
|---|---|---|
| Optimal k by intrinsic metrics | K-Means and GMM, k ∈ [2, 15]: inertia (elbow), Silhouette, Davies-Bouldin (+ BIC for GMM) | `clustering_elbow.png`, `clustering_silhouette.png`, `clustering_davies_bouldin.png` |
| Agreement with ground truth | ARI, NMI, Purity (`src.clustering.purity_score`) | `clustering_benchmark.csv` |
| Raw vs latent space | Same sweep on raw pixels, PCA-50, PCA-90%, PCA-95% (PCA fit on train) | `clustering_benchmark.csv`, `clustering_space_comparison.png` |
| Interpretation | Contingency heatmap cluster × class at k=10 | `cluster_class_contingency.png` |

## C. Classification — `experiments/run_classification.py`

| Requirement | Method | Output |
|---|---|---|
| Linear vs non-linear | Logistic Regression, Linear SVM vs RBF-SVM, Random Forest (CNN if brief requires) | `classification_benchmark.csv` |
| Dimensionality vs accuracy and efficiency | Each model on raw and PCA-k (k ∈ {10, 20, 50, 100, 90%, 95%}); accuracy, macro-F1, fit time, inference time | `classification_benchmark.csv`, `accuracy_vs_dimension.png`, `inference_time_vs_dimension.png` |
| Hyperparameter selection | `Pipeline(StandardScaler → PCA → model)`, grid search on train, chosen on val | `model_selection.csv` |
| Ambiguous classes and error analysis | Row-normalised confusion matrices (val), most-confused class pairs | `confusion_matrix_<model>.png`, `confusion_matrix_<model>.csv`, `top_confusions.csv` |

## D. Generalisation — `experiments/run_classification.py`

| Requirement | Method | Output |
|---|---|---|
| Learning curves | Train on 10/25/50/75/100% of train (seeded stratified subsets); score on that subset and on val | `learning_curves.csv`, `learning_curve_<model>.png` |
| Generalisation gap | \|train − val\| per fraction and model | `learning_curves.csv`, `generalisation_gap.png` |
| Bias vs variance diagnosis and fix | Regularisation sweeps (LogReg/SVM `C`, RF `max_depth`, PCA k) as validation curves | `validation_curve_<model>.png`, `validation_curves.csv` |
| Single out-of-sample test evaluation | `--final` refits selected models on train, scores test once | `test_results.json` |

## Phase 5: audit — `tests/test_experiments.py`

- [x] Split disjoint, stratified, deterministic; shapes and ranges of the loaded data
- [x] Purity metric correctness
- [ ] Leakage: fitted scaler/PCA statistics equal train-only statistics
- [ ] Determinism: each runner gives identical metrics on two runs with `--seed 42`
- [ ] All figures saved at 300 DPI
- [ ] Static check: no `fit` on val/test; every stochastic estimator receives `random_state=seed`

## Definition of done (from directive §5)

- [ ] Every mandatory question in the brief answered with empirical data
- [ ] All required plots in `outputs/figures/` at 300 DPI
- [ ] Accuracy, Macro F1, ARI, NMI, Silhouette, Explained Variance logged in `outputs/metrics/`
- [ ] Seeds fixed to 42; pipeline runs end-to-end without errors
- [ ] Zero leakage: preprocessors and projections fit on train only
