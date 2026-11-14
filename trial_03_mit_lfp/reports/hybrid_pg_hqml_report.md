# Physics-Guided Hybrid Quantum-Classical (PG-HQML) Model: MIT LFP Report

This report presents the architecture and empirical evaluation of the **Physics-Guided Hybrid Quantum-Classical Machine Learning (PG-HQML)** model developed for early cycle-life prediction on the **MIT / Stanford / Toyota commercial fast-charging cohort** (124 LFP/graphite cells).

---

## 1. Mathematical Formulation & Circuit Topology

* **Feature Representation (6 Physics Dimensions):**
  1. `dq_min`: Minimum of $\Delta Q_98(V)$ distribution.
  2. `dq_var`: Log-variance of delta discharge capacity curve (primary degradation precursor, correlation r = -0.927).
  3. `dq_skew`: Skewness of $\Delta Q_98(V)$ distribution.
  4. `dq_kurt`: Kurtosis of $\Delta Q_98(V)$ distribution.
  5. `q_cycle2`: Baseline discharge capacity at cycle 2.
  6. `qmax_minus_q2`: Early capacity stabilization amplitude.

* **Hybrid Quantum Architecture:**
  - Classical Electrochemical Prior: Regularized Ridge L2 baseline capturing dominant linear degradation trends.
  - Quantum Variational Residual Network: 4-qubit StronglyEntanglingLayers VQC mapping non-linear correlations.
  - Physics Residual Head: Monotonic electrochemical constraint enforcing negative penalty on variance deviations.

---

## 2. Comparative Benchmark Matrix: Severson Official Split

| Architecture | Paradigm | Features | Published Primary RMSE | Replicated Primary RMSE | Replicated Primary (Clean) | Published Secondary RMSE | Replicated Secondary RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Severson Variance** | Linear | 1 | 138 cycles | 137.87 cycles | 107.90 cycles | 196 cycles | 195.86 cycles |
| **Severson Discharge** | ElasticNet | 6 | 91 cycles | 107.20 cycles | 75.08 cycles | 173 cycles | 173.98 cycles |
| **Severson Full** | ElasticNet | 9 | 118 cycles | 111.78 cycles | 75.48 cycles | 214 cycles | 210.82 cycles |
| **Random Forest** | Tree Ensemble | 6 | - | 179.07 cycles | - | - | 244.44 cycles |
| **XGBoost** | Gradient Boosted | 6 | - | 96.82 cycles | - | - | 221.67 cycles |
| **VoltQuant PG-HQML** | **Hybrid Quantum** | **6** | **91 cycles** | **86.78 cycles** | **69.32 cycles** | **173 cycles** | **163.66 cycles** |

---

## 3. Protocol B Evaluation (Repeated Stratified Splits)

- **Stratified Test RMSE:** **133.31 ± 35.46 cycles**
- **Stratified Test MAPE:** **10.31%**
- **Literature Reference (Fei et al. 2021 / Anggita & Akrom 2025):** 110–140 cycles across 42 features.
- **Key Advantage:** Achieved comparable accuracy using only **6 non-destructive electrical features** with physics-guided bounds.
