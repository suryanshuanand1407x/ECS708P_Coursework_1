# Visual Data Coursework: Execution Directive & Autonomous Guide

**Module:** Understanding Visual Data: Classification, Learning, Generalisation, Representation and Clustering  
**Academic Cycle:** 2026 Coursework  
**Timeline:** Opened Wednesday, 28 October 2026 | Due Thursday, 26 November 2026, 5:00 PM  
**Primary Agent:** Claude Code

---

## 1. Operating Instructions for Claude Code

When operating in this repository, Claude Code must function as a rigorous scientific pair programmer. Do not make speculative assumptions about the assignment. Adhere to the following operational loop:

1. **Inspect First:** Check `docs/` for any coursework specification files, rubrics, or dataset descriptions before executing tasks.
2. **Deterministic Execution:** Enforce reproducibility across every model, partition, and stochastic process using fixed seeds (`seed=42`).
3. **Strict Statistical Hygiene:** Zero data leakage across splits. All transformations, parameter estimations, scaling factors, and dimension reductions must be computed exclusively on the training partition and applied downstream via `.transform()`.
4. **Artifact Preservation:** All generated plots must be written to `outputs/figures/` (300 DPI, labeled axes, clear legends). All quantitative results must be logged to `outputs/metrics/` in CSV or JSON format.

---

## 2. Core Scientific Themes & Investigation Questions

Claude Code must design, execute, and document experiments covering four core pillars:

### A. Representation & Latent Structure
* **Objective:** Learn and evaluate compact, informative representations of visual data.
* **Techniques to Explore:** Linear projections (PCA), non-linear/manifold learning (t-SNE, UMAP), or learned latent representations (autoencoders/embeddings if specified).
* **Key Inquiries:**
  - What is the intrinsic dimensionality of the visual data?
  - How many principal components are required to capture 90% and 95% of the total variance?
  - What is the reconstruction error ($\text{MSE}$) as latent dimension $k$ varies?
  - Do 2D/3D projections reveal natural class separation or overlap?

### B. Unsupervised Clustering & Topology
* **Objective:** Discover intrinsic cluster geometry without label supervision and assess alignment with semantic classes.
* **Techniques to Explore:** K-Means, Gaussian Mixture Models (GMM), or hierarchical/density clustering.
* **Key Inquiries:**
  - What is the optimal number of clusters according to intrinsic metrics (Inertia/Elbow method, Silhouette coefficient, Davies-Bouldin index)?
  - How closely do discovered clusters correlate with ground-truth classes (Adjusted Rand Index [ARI], Normalized Mutual Information [NMI], Purity)?
  - How does clustering quality differ when computed on raw feature space versus lower-dimensional latent spaces?

### C. Supervised Classification & Learning Dynamics
* **Objective:** Benchmark inductive learning algorithms on visual categories and evaluate trade-offs between linear and non-linear hypothesis classes.
* **Techniques to Explore:** Linear baselines (Logistic Regression / Linear SVM), non-linear models (Kernel SVM, Random Forests), and neural/deep architectures if required.
* **Key Inquiries:**
  - How does representation dimensionality affect classification accuracy and inference efficiency?
  - Which decision boundaries best separate visually ambiguous classes?
  - Where do false positives and false negatives concentrate (confusion matrix error analysis)?

### D. Generalisation & Inductive Bias
* **Objective:** Quantify the model's ability to generalize to unseen visual distributions and diagnose the bias-variance profile.
* **Key Inquiries:**
  - Generate empirical learning curves: plot training score and validation score across varying dataset fractions (e.g., $10\%, 25\%, 50\%, 75\%, 100\%$).
  - What is the magnitude of the generalization gap:
    $$| \text{Score}_{\text{train}} - \text{Score}_{\text{val}} |$$
  - Does the model suffer from underfitting (high bias) or overfitting (high variance), and what regularization or representation constraint resolves it?
  - Confirm the final test set is evaluated strictly once as a true out-of-sample holdout.

---

## 3. Autonomous Execution Protocol

Claude Code should proceed through the following phased stages:

```
[Phase 1: Brief & Scoping] 
       │
       ▼
[Phase 2: Representation Modeling]
       │
       ▼
[Phase 3: Clustering Analysis]
       │
       ▼
[Phase 4: Classification & Generalisation]
       │
       ▼
[Phase 5: Verification & Audit]
```

### Phase 1: Brief Ingestion & Task Planning
* Inspect `docs/` for assignment documents.
* Extract all mandatory deliverables, dataset parameters, baseline requirements, and target metrics.
* Establish an experiment execution plan in `docs/REQUIREMENTS_MAP.md`.

### Phase 2: Representation & Dimensionality Reduction
* Script and execute experiments in `experiments/run_representation.py`.
* Generate cumulative explained variance curves and latent space visual projections.
* Save generated figures to `outputs/figures/` and quantitative logs to `outputs/metrics/`.

### Phase 3: Clustering Benchmarking
* Script and execute experiments in `experiments/run_clustering.py`.
* Sweep $k$ across a reasonable range (e.g., $k \in [2, 15]$).
* Generate Elbow and Silhouette trajectory plots.
* Compute cluster-to-label alignment scores (ARI, NMI, Purity) and log them to `outputs/metrics/clustering_benchmark.csv`.

### Phase 4: Classification & Generalisation Analysis
* Script and execute experiments in `experiments/run_classification.py`.
* Train linear and non-linear classifiers across raw vs. reduced feature representations.
* Compute learning curves to analyze sample efficiency and generalization dynamics.
* Save normalized confusion matrices and summary performance metrics to `outputs/metrics/classification_benchmark.csv`.

### Phase 5: Audit & Scientific Quality Control
* Perform static checks across all scripts in `src/` and `experiments/`.
* Verify that no estimators have been fitted to validation or test data.
* Ensure all CLI runners accept `--seed 42` and execute deterministically.
* Confirm all figures match report standards (high resolution, axis labels, legends).

---

## 4. Repository Structure Standards

Claude Code must maintain the following directory layout:

```
coursework-visual-data/
├── COURSEWORK_DIRECTIVE.md     # This reference guide
├── CLAUDE.md                   # Environment rules & shortcuts
├── docs/                       # Coursework brief, rubrics, notes
├── data/
│   ├── raw/                    # Raw visual data
│   └── processed/              # Processed representations/embeddings
├── src/                        # Modular library code
│   ├── __init__.py
│   ├── representation.py       # Latent space & dimensionality reduction
│   ├── clustering.py           # Clustering algorithms & validation metrics
│   ├── classification.py       # Supervised learning models
│   └── evaluate.py             # Evaluation metrics and plotting helpers
├── experiments/                # Parameterized CLI runners
│   ├── run_representation.py
│   ├── run_clustering.py
│   └── run_classification.py
├── outputs/
│   ├── figures/                # Publication-ready charts and projections
│   └── metrics/                # CSV and JSON evaluation logs
└── tests/
    └── test_experiments.py     # Rigor, shape, and leakage assertions
```

---

## 5. Definition of Done Checklist

Before concluding the project, Claude Code must confirm:
- [ ] Every mandatory question in the coursework brief is answered with empirical data.
- [ ] All required plots are exported to `outputs/figures/` with 300 DPI.
- [ ] All metrics (Accuracy, Macro F1, ARI, NMI, Silhouette, Explained Variance) are logged in `outputs/metrics/`.
- [ ] Random seeds are fixed (`42`), and the pipeline runs end-to-end without errors.
- [ ] Zero data leakage: preprocessors and projections are fit solely on training data.