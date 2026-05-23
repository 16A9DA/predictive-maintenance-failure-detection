# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Predictive maintenance classifier using the [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset). The goal is binary classification: predict `Machine failure` (0/1) from machine sensor readings.

## Running the Notebook

```bash
jupyter notebook notebooks/notebook.ipynb
# or
jupyter lab notebooks/notebook.ipynb
```

> **Note**: The dataset path is hardcoded as an absolute path in Cell 1. Update it to match your local path:
> ```python
> df = pd.read_csv('/Users/ryn/Documents/riskpred/data/ai4i2020.csv')
> ```

## Architecture & Data Flow

The entire workflow lives in `notebooks/notebook.ipynb`. It follows this sequence:

1. **Load & inspect** — reads `data/ai4i2020.csv` (10,000 rows, 14 columns)
2. **Drop identifier columns** — `UDI`, `Product ID` removed; they carry no predictive signal
3. **EDA / visualisation** — scatter plots of 4 sensor features vs `Machine failure`, saved as PNGs in `notebooks/`
4. **Drop failure-mode sub-labels** — `TWF`, `HDF`, `PWF`, `OSF`, `RNF` are removed; the model predicts the aggregate `Machine failure` flag only, not the individual failure modes
5. **Preprocessing pipeline** — `StandardScaler` on numeric features; `OneHotEncoder(handle_unknown="ignore")` on `Type` (L / M / H product quality tiers)
6. **Model** — `LogisticRegression(max_iter=1000)` inside a sklearn `Pipeline`; 80/20 train-test split with `random_state=42`

## Dataset Schema

| Column | Role | Notes |
|--------|------|-------|
| `Type` | Feature (categorical) | L / M / H product quality tier |
| `Air temperature [K]` | Feature (numeric) | |
| `Process temperature [K]` | Feature (numeric) | |
| `Rotational speed [rpm]` | Feature (numeric) | |
| `Torque [Nm]` | Feature (numeric) | |
| `Tool wear [min]` | Feature (numeric) | |
| `Machine failure` | **Target** | Binary; ~3.4% positive rate → class imbalance |
| `TWF`, `HDF`, `PWF`, `OSF`, `RNF` | Dropped | Individual failure-mode flags (subset of `Machine failure`) |

## Key Design Notes

- **Class imbalance**: Only ~3.4% of samples are failures. Logistic Regression with default settings will bias toward predicting 0. When extending the model, consider `class_weight='balanced'`, SMOTE, or switching to a tree-based model, and evaluate with precision/recall/F1 rather than accuracy alone.
- **Failure sub-labels**: TWF (tool wear), HDF (heat dissipation), PWF (power), OSF (overstrain), RNF (random) are intentionally excluded from features to prevent data leakage — they are derived from the same failure event.
- **Plot output paths**: `plt.savefig()` calls in the notebook use relative paths; plots are saved relative to wherever the notebook kernel is started (typically `notebooks/`).
