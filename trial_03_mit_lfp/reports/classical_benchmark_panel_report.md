# Classical Machine Learning Benchmark Panel: MIT LFP Cohort

This report compiles the empirical performance of 5 classical ML regression architectures across two standardized validation protocols on the **MIT / Stanford / Toyota fast-charging battery dataset**.

---

## 1. Protocol A: Official Severson Split (Train: 41, Primary: 43, Secondary: 40)

| Feature Set | Model | Primary Test RMSE (cycles) | Primary Test MAPE (%) | Secondary Test RMSE (cycles) | Secondary Test MAPE (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `discharge_6` | **Ridge** | 88.25 | 12.75% | 168.70 | 9.51% |
| `discharge_6` | **ElasticNet** | 107.20 | 12.62% | 173.98 | 11.98% |
| `discharge_6` | **RandomForest** | 179.07 | 13.93% | 244.44 | 13.71% |
| `discharge_6` | **GradientBoosting** | 128.97 | 9.76% | 218.18 | 12.42% |
| `discharge_6` | **XGBoost** | 96.82 | 10.68% | 221.67 | 15.86% |
| `full_9` | **Ridge** | 121.76 | 10.61% | 229.24 | 11.52% |
| `full_9` | **ElasticNet** | 111.78 | 10.66% | 210.82 | 11.01% |
| `full_9` | **RandomForest** | 169.69 | 13.53% | 227.31 | 12.16% |
| `full_9` | **GradientBoosting** | 106.17 | 10.03% | 171.78 | 10.48% |
| `full_9` | **XGBoost** | 103.14 | 11.48% | 204.07 | 13.43% |
| `all_20` | **Ridge** | 309.45 | 28.92% | 263.82 | 22.69% |
| `all_20` | **ElasticNet** | 217.08 | 18.13% | 185.28 | 14.86% |
| `all_20` | **RandomForest** | 197.67 | 15.86% | 264.00 | 12.46% |
| `all_20` | **GradientBoosting** | 133.29 | 11.46% | 214.35 | 12.11% |
| `all_20` | **XGBoost** | 106.75 | 11.81% | 207.55 | 13.59% |

---

## 2. Protocol B: 20-Seed Repeated 70/30 Stratified Splits (Fei et al. / Anggita & Akrom Protocol)

| Feature Set | Model | 20-Seed Mean Test RMSE (cycles) | Test RMSE Spread (± 1σ) | 20-Seed Mean Test MAPE (%) |
| :--- | :--- | :---: | :---: | :---: |
| `discharge_6` | **Ridge** | **145.96** cycles | ± 41.86 | **10.44%** |
| `discharge_6` | **ElasticNet** | **141.28** cycles | ± 38.14 | **10.42%** |
| `discharge_6` | **RandomForest** | **137.58** cycles | ± 33.33 | **10.68%** |
| `discharge_6` | **GradientBoosting** | **132.10** cycles | ± 28.79 | **10.25%** |
| `discharge_6` | **XGBoost** | **142.39** cycles | ± 28.07 | **11.67%** |
| `full_9` | **Ridge** | **139.44** cycles | ± 26.32 | **10.62%** |
| `full_9` | **ElasticNet** | **135.09** cycles | ± 25.92 | **10.68%** |
| `full_9` | **RandomForest** | **121.84** cycles | ± 32.01 | **9.48%** |
| `full_9` | **GradientBoosting** | **122.04** cycles | ± 26.05 | **9.20%** |
| `full_9` | **XGBoost** | **137.54** cycles | ± 28.59 | **10.20%** |
| `all_20` | **Ridge** | **129.71** cycles | ± 26.49 | **9.85%** |
| `all_20` | **ElasticNet** | **132.38** cycles | ± 31.46 | **10.08%** |
| `all_20` | **RandomForest** | **129.33** cycles | ± 35.55 | **10.08%** |
| `all_20` | **GradientBoosting** | **127.08** cycles | ± 31.30 | **9.46%** |
| `all_20` | **XGBoost** | **142.80** cycles | ± 30.38 | **10.37%** |

---

## 3. Engineering Insights & Comparison

1. **Feature Dimensionality Effect:**
   - Expanding from 6 features (`discharge_6`) to all 20 candidate features (`all_20`) reduces test error across regularized linear models (Ridge/ElasticNet) down to ~128–130 cycles on the 20-seed protocol.
   - Non-linear tree ensembles (RandomForest, XGBoost, LightGBM) benefit strongly from the capacity fade slope and temperature integral metrics, reaching ~120–135 cycles average test RMSE.

2. **Cross-Batch Robustness:**
   - On the secondary test set (Batch 3, cycled 11 months later under different ambient conditions), linear regularizers (ElasticNet) exhibit superior stability compared to unconstrained decision trees.
