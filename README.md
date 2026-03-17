# 🏥 Insurance Charges Prediction — Support Vector Regression (SVR)

A machine learning project that predicts medical insurance charges using **Support Vector Regression (SVR)** with an end-to-end `scikit-learn` Pipeline, including preprocessing, feature selection, and model evaluation.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Results](#results)
- [Feature Importance](#feature-importance)
- [Requirements](#requirements)
- [Usage](#usage)

---

## Overview

This notebook builds a regression model to predict medical insurance charges based on patient demographics and lifestyle information. It uses **Support Vector Regression (SVR)** with an RBF kernel, wrapped in a clean `scikit-learn` Pipeline that handles encoding, scaling, and prediction in a single flow.

A key highlight of this project is the **iterative permutation-importance-based feature selection** with early stopping, which automatically identifies the optimal subset of features.

---

## Dataset

**Source:** `insurance.csv`

| Feature    | Type        | Description                          |
|------------|-------------|--------------------------------------|
| `age`      | Numerical   | Age of the primary beneficiary       |
| `sex`      | Categorical | Gender of the beneficiary            |
| `bmi`      | Numerical   | Body Mass Index                      |
| `children` | Numerical   | Number of dependents covered         |
| `smoker`   | Categorical | Smoking status (yes/no)              |
| `region`   | Categorical | Residential region in the US         |
| `charges`  | Numerical   | ⭐ **Target** — Individual medical costs |

- **Total records:** 1,338 (1 duplicate removed → 1,337 used)
- **Missing values:** None

---

## Project Workflow

### 1. Exploratory Data Analysis (EDA)
- Inspected data types, shape, and missing values
- Visualized the distribution of `charges` — identified **right-skew** (non-normal)
- Created a scatter plot of `age` vs `charges` colored by `smoker` status to understand the charge pattern
- Investigated high-charge records (>$30,000) and confirmed they are justified by known risk factors (smoking, high BMI, older age) — **not treated as outliers**

### 2. Preprocessing
- Separated features (`X`) and target (`y = charges`)
- Auto-detected categorical columns (`sex`, `smoker`, `region`) for One-Hot Encoding
- Split data: **80% train / 20% test** (`random_state=0`)

### 3. Model Training (sklearn Pipeline)
```
Pipeline:
  └── ColumnTransformer (OneHotEncoder for categoricals, passthrough for numericals)
  └── StandardScaler
  └── SVR (kernel='rbf')
```
- Target variable (`y`) was also scaled using a separate `StandardScaler` before training
- Predictions were inverse-transformed back to original dollar scale

### 4. Evaluation (All Features)
| Metric | Value  |
|--------|--------|
| R²     | 0.8403 |
| MAE    | $2,881 |
| MAPE   | 0.247  |
| RMSE   | $5,183 |
| MAPE (mean-normalised) | 0.205 |

### 5. Feature Importance (Permutation Importance)
Used `sklearn.inspection.permutation_importance` on the scaled target to rank features by their contribution to model performance.

| Rank | Feature    | Importance |
|------|------------|------------|
| 1    | `smoker`   | 1.226303   |
| 2    | `bmi`      | 0.217089   |
| 3    | `age`      | 0.179217   |
| 4    | `children` | 0.010607   |
| 5    | `region`   | 0.005846   |
| 6    | `sex`      | -0.000716  |

### 6. Iterative Feature Selection
- Looped through importance thresholds (low → high) to progressively remove low-importance features
- Retrained the pipeline at each threshold and tracked R²
- Applied **early stopping** (patience = 6 rounds without improvement)

**Best Result:** Dropping `sex` (importance < 0) improved the model:

| Configuration         | R²     |
|-----------------------|--------|
| All 6 features        | 0.8403 |
| Best 5 features (no `sex`) | **0.8416** |
| R² Improvement        | +0.0013 |

---

## Results

The final model uses **5 features** (`smoker`, `bmi`, `age`, `children`, `region`) and achieves:

- **R² = 0.8416** on the test set
- `smoker` is by far the most dominant predictor of insurance charges

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
statsmodels
scikit-learn
```

Install all dependencies:
```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn
```

---

## Usage

1. Clone the repository and place `insurance.csv` in the same directory as the notebook.
2. Open `Insurance_SVR.ipynb` in Jupyter or JupyterLab.
3. Run all cells from top to bottom.

```bash
jupyter notebook Insurance_SVR.ipynb
```

---

## 📝 Notes

- The `charges` distribution is right-skewed due to the strong influence of smoking status on costs — this is a natural characteristic of the data, not a data quality issue.
- High-charge records (>$30,000) were retained as they correspond to legitimate high-risk profiles (smokers, high BMI, older age).
- SVR requires feature scaling — both `X` and `y` are scaled before training to ensure optimal SVM performance.
