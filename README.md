# Interpretable ML: Telco Customer Churn (Regression + Classification + GAM)

## Overview
- Goal: Build interpretable models for Telco customer churn and rigorously check regression assumptions.
- Dataset: `WA_Fn-UseC_-Telco-Customer-Churn.csv` (Telco churn sample dataset). Target is `Churn` (converted to `Churn_Yes`).
- Notebook: `regression_interpretability.ipynb` walks from data prep → diagnostics → linear/logistic baselines → Lasso regularization → Generalized Additive Model (GAM) → model comparison.

## Contents
- `regression_interpretability.ipynb`: Main analysis and models.
- `WA_Fn-UseC_-Telco-Customer-Churn.csv`: Source data (in repo root).

## Environment
- Python 3.9+ recommended.
- Key packages:
  - pandas, numpy, matplotlib, seaborn
  - statsmodels, scipy
  - scikit-learn
  - pygam

Install everything (within an activated virtual env):

```bash
pip install -U pip
pip install pandas numpy matplotlib seaborn statsmodels scipy scikit-learn pygam
```

If `pygam` build fails on Linux, ensure build tools are available:

```bash
sudo apt-get update && sudo apt-get install -y build-essential python3-dev
```

## Quickstart
- VS Code: Open this folder and run cells in `regression_interpretability.ipynb` with the Python kernel.
- Jupyter (CLI):

```bash
jupyter lab  # or: jupyter notebook
```

The notebook expects the CSV at the project root (same folder as the notebook).

## What the Notebook Does
1. Data loading and cleaning
   - Drops `customerID` (identifier).
   - Coerces `TotalCharges` to numeric and drops empty rows.
   - One‑hot encodes categoricals (with reference levels to avoid dummy trap).
2. Exploratory checks and assumptions
   - Linearity: scatter/residual plots; Ramsey RESET.
   - Independence: Durbin–Watson, ACF/PACF, Breusch–Godfrey.
   - Homoscedasticity: residuals vs fitted; Breusch–Pagan, White, Goldfeld–Quandt.
   - Normality: histogram, Q–Q; Shapiro/K–S/Anderson/Jarque–Bera.
   - Multicollinearity: VIF.
   - Influence: Cook’s distance, studentized residuals, DFBETAS.
3. Models
   - Linear regression with robust (HC3) SEs for illustration.
   - Logistic regression (with and without L1 via LogisticRegressionCV).
   - Lasso (L1) for feature selection on a standardized pipeline.
   - GAM (LogisticGAM) with splines for continuous features and factors for categoricals; grid search over λ.
4. Comparison and interpretation
   - ROC‑AUC and summaries; interpretability via odds ratios and GAM partial effects.

## Key Observations (as implemented in the notebook)
- Linearity is violated; RESET test rejects correct specification → suggests non‑linearities/interactions.
- Independence looks fine; no material autocorrelation detected.
- Heteroscedasticity is present (Breusch–Pagan/White reject H0).
- Residuals deviate from normality (multiple tests reject normality).
- Notable collinearity: `MonthlyCharges` vs `TotalCharges`; moderate relationships with `tenure`.
- Influence diagnostics show no problematic high‑influence outliers.
- Performance: GAM attains the highest ROC‑AUC in‑notebook; L1‑regularized logistic is a close, more parsimonious and highly interpretable alternative.

## Reproducing Results / Tips
- Randomness: The notebook sets `random_state=42` for splits/regularization where applicable, aiding reproducibility.
- Standardization: Pipelines standardize predictors for L1 models; keep this when changing features.
- GAM notes: Reported p‑values can be unreliable due to smoothing estimation; rely on shapes and cross‑validated performance.
- Multicollinearity: Consider dropping either `TotalCharges` or `MonthlyCharges` for simpler linear models.

## Troubleshooting
- `pygam` install errors: ensure system compilers are installed (see above). Try `pip install --no-build-isolation pygam` if needed.
- Backend issues when plotting in some environments: set `matplotlib` backend, e.g. `import matplotlib; matplotlib.use("Agg")` when running headless.
- Memory/time: GAM grid search over many terms can be slow; reduce λ grid or terms for quick runs.

## Citation / Data Source
- Telco Churn sample dataset is widely attributed to IBM Sample Data (a common teaching dataset). Ensure usage aligns with your course or project requirements.

## License
- No explicit license is provided in this repository. If you plan to reuse code or data, please add an appropriate license and verify dataset terms.
