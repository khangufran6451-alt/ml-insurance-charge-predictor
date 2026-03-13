# 📊 Regression Model with Automated Feature Selection

## 🚀 Overview

This project demonstrates a **regression workflow** using automated **feature selection** to improve model performance.
It covers:

* Baseline evaluation with **all features**
* **Permutation importance** for feature contribution
* Aggregation of **encoded features**
* **Iterative feature selection** with **best threshold** and early stopping
* Final model retraining and evaluation

---

## 🛠️ Features

* **Baseline Model Evaluation** – Train and evaluate model with all features
* **Permutation Feature Importance** – Identify most important features
* **Aggregated Feature Importance** – Group importance by original features
* **Iterative Feature Selection** – Automatically find the **best threshold**
* **Final Model Retraining** – Train using only selected optimal features
* **Evaluation** – Compare predicted vs actual values, R² scores

---

## 📦 Requirements

```bash
Python 3.x
pip install numpy pandas scikit-learn matplotlib seaborn
```

---

## ⚡ Usage

1. **Load dataset**

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

2. **Train baseline model**

```python
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
```

3. **Compute permutation importance**

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(regressor, X_test, y_test, n_repeats=20, random_state=42)
importance = result.importances_mean
```

4. **Aggregate and group feature importance**

```python
group_importance = importance_df.groupby("Original_Feature")["Importance"].mean()
```

5. **Iterative feature selection with best threshold**

```python
# Automatically select best threshold and retrain model
```

6. **Final evaluation**

```python
y_pred_final = regressor.predict(X_test_final)
r2_final = r2_score(y_test, y_pred_final)
```

---

## 📊 Results

* **Baseline R² Score (All Features):** `baseline_r2`
* **Final R² Score (Selected Features):** `final_r2`
* **Number of Selected Features:** `num_features`

> Visualizations of feature importance and predicted vs actual values can be added using `matplotlib` or `seaborn`.

---

## 🗂 Project Structure

```text
├── data/                  # Dataset files
├── notebooks/             # Jupyter notebooks
├── src/                   # Python scripts for modeling & feature selection
├── README.md              # Project overview
└── requirements.txt       # Dependencies
```

---

## 🤝 Contributing

* You may **view or copy this code** for personal, educational, or professional use.
* You **may NOT modify, redistribute, or create derivative works** without explicit permission from the author.

---

## 📄 License

```text
Copyright (c) 2026 Mohammad Ghufran Khan

All rights reserved.

You may view and copy this code for personal or educational use only.
You may NOT modify, distribute, or create derivative works based on this code
without explicit permission from the author.
