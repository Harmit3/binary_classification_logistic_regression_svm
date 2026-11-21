# Foundations of Machine Learning : Classification & Kernel Methods

An experimental study of binary classification using logistic regression and Support Vector Machines (SVMs), accompanied by mathematical analysis of probabilistic models, margins and kernel validity. 

## Project overview

This project explores how linear and nonlinear classifiers fit a two-dimensional dataset. It implements unregularized logistic regression using numerical optimization, compares three SVM kernels, searches hyperparameters, visualizes decision boundaries and evaluates optimization convergence.

The associated theoretical coursework covers:

- Generative models, Naive Bayes, conditional and posterior probabilities, Bayes' theorem and Laplace smoothing.
- SVM margins, support vectors, hinge loss and the effect of removing training observations.
- Kernel validity, feature mappings and positive-semidefinite Gram matrices, including a counterexample showing that a weighted difference of valid kernels need not be valid.

## Technologies

Python, Jupyter Notebook, NumPy, pandas, Matplotlib, SciPy, scikit-learn and IPython. The recorded notebook environment uses Python 3.14.0 and scikit-learn 1.8.0.

## Dataset

`Q4/cls.dat` is the course-provided dataset. It contains 800 observations, two numerical features and a binary class label. The first row is skipped during loading. The notebook checks the shape, finite values and label domain before training.

| Class | Observations | Plot color |
|---|---:|---|
| 0 | 377 | Blue |
| 1 | 423 | Red |
| Total | 800 | |

All models use the original feature coordinates without scaling.

![Training observations](Q4/q4a_data.png)

## Methodology

### 1. Logistic regression

The notebook implements mean binary cross-entropy using a numerically stable `numpy.logaddexp` expression and an analytical gradient. SciPy's BFGS optimizer estimates two weights and an intercept without L2 regularization. The decision boundary corresponds to a predicted probability of 0.5.

The recorded optimization converged with mean training log loss 0.619533 and gradient infinity norm approximately 9.41 × 10⁻¹².

### 2. Support Vector Machines

Three `sklearn.svm.SVC` models are compared:

- **Linear:** dot-product kernel; searches five values of C (0.01, 0.1, 1, 10, 100).
- **Gaussian (RBF):** exp(−γ‖x−z‖²); searches 45 combinations of C and gamma.
- **Polynomial:** (γ xᵀz + coef0)^degree; searches 152 configurations across C, gamma, polynomial degree and coef0, including a broader high-degree search.

The search selects the first converged candidate attaining the highest observed training accuracy within each grid. SVC uses tolerance 1e-4 and a maximum of 2,000,000 iterations. Some polynomial candidates reach the iteration limit and are excluded from selection.

### 3. Evaluation and visualization

The notebook reports training accuracy, correct predictions, confusion matrices, support-vector counts, solver status and iteration counts. It also computes mean hinge loss and the primal objective using the kernel expansion. Decision boundaries are plotted as zero-score contours on a 400 × 400 grid.

## Results

The following results are from the saved notebook outputs, not a new independent run.

| Model | Selected parameters | Training accuracy | Correct / 800 | Support vectors |
|---|---|---:|---:|---:|
| Logistic regression | No regularization | 67.000% | 536 | — |
| Linear SVM | C=10 | 70.875% | 567 | 579 |
| Gaussian SVM | C=100, gamma=5 | 100.000% | 800 | 339 |
| Polynomial SVM | C=10000, gamma=0.1, degree=8, coef0=1 | 100.000% | 800 | 30 |

### Confusion matrices

Rows represent actual labels; columns represent predicted labels, ordered 0 and 1.

| Model | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Logistic regression | 234 | 143 | 121 | 302 |
| Linear SVM | 258 | 119 | 114 | 309 |
| Gaussian SVM | 377 | 0 | 0 | 423 |
| Polynomial SVM | 377 | 0 | 0 | 423 |

### Optimization diagnostics

| Model | Mean hinge loss | Primal objective |
|---|---:|---:|
| Linear SVM | 0.721920490 | 5775.445538 |
| Gaussian SVM | 0.000507125 | 356.286220 |
| Polynomial SVM | 0.000574850 | 16658.776226 |

The selected SVMs converged. Primal objectives use different C values and feature spaces, so their magnitudes should not be treated as directly comparable performance scores.

## Decision boundary visualizations

### Linear models

| Logistic regression | Linear SVM |
|---|---|
| ![Logistic regression](Q4/q4b_logistic.png) | ![Linear SVM](Q4/q4c_linear_svm.png) |

### Nonlinear models

| Gaussian SVM | Polynomial SVM |
|---|---|
| ![Gaussian SVM](Q4/q4d_gaussian_svm.png) | ![Polynomial SVM](Q4/q4e_polynomial_svm.png) |

## Key findings and limitations

- Nonlinear kernels fit the training observations substantially better than the linear models in this experiment.
- Both nonlinear models achieve perfect training classification; the selected polynomial model uses fewer support vectors than the selected Gaussian model.
- Hyperparameters are selected on training accuracy, with no held-out test set or cross-validation. Perfect training accuracy is not evidence of perfect performance on unseen data.
- The polynomial search can be computationally expensive and some configurations do not converge within the iteration limit.
- Future improvements include stratified cross-validation, held-out testing, feature scaling experiments, runtime benchmarking and regularization analysis.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
└── Q4/
    ├── Assn2_Q4.ipynb
    ├── cls.dat
    ├── q4a_data.png
    ├── q4b_logistic.png
    ├── q4c_linear_svm.png
    ├── q4d_gaussian_svm.png
    └── q4e_polynomial_svm.png
```

## Getting started

From the repository root, install the dependencies in a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Then run:

```bash
python -m pip install -r requirements.txt
cd Q4
python -m jupyter notebook Assn2_Q4.ipynb
```

Place an authorized copy of `cls.dat` in `Q4/` if it is not distributed with the repository. Run all notebook cells in order. The notebook regenerates the five PNG files in its working directory and displays model metrics and diagnostics. Exact results and runtime may vary with library versions and hardware.

## Skills demonstrated

Mathematical modeling, binary classification, numerical optimization, kernel methods, hyperparameter search, model evaluation, convergence analysis, scientific Python, data visualization and reproducible computational notebooks.

## Author and acknowledgments

**Harmit Patel** — Foundations of Machine Learning coursework, Western University. Course dataset and instructional materials are credited to their respective owners. This repository is an educational project; redistribution of course materials and data is subject to the applicable permissions. No open-source license is asserted for third-party materials.
