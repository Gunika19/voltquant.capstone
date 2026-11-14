# Trial 3: MIT / Stanford Commercial Cohort (LFP / Graphite Fast-Charging)

This module implements the data ingestion, baseline reproduction, and physics-guided hybrid machine learning models for early cycle-life prediction on the **MIT / Stanford / Toyota Research Institute** fast-charging dataset (Severson et al., *Nature Energy*, 2019).

---

## 1. Scientific Objective & Dataset Scope

Predicting the end-of-life cycle count of commercial lithium-ion cells from the first 100 cycles enables rapid manufacturing quality screening, accelerated battery development, and optimal second-life sorting.

* **Cell Specifications:** Commercial A123 APR18650M1A cylindrical cells (Nominal capacity: 1.1 Ah, nominal voltage: 3.3 V).
* **Cathode/Anode Chemistry:** Lithium Iron Phosphate ($\text{LiFePO}_4$, LFP) / Graphite.
* **Testing Conditions:** 72 distinct fast-charging policies (e.g., reaching 80% SOC in under 10–20 minutes), all discharged at a standard 4C rate down to 2.0 V.
* **Failure Definition:** EOL is reached when cell capacity drops to 80% of nominal rating (0.88 Ah). Lifespans range from **148 to 2,237 cycles**.
* **Cohort Size:** 124 cells across 3 production batches:
  * Batch 1 (`b1`): 41 cells (Severson Training cohort)
  * Batch 2 (`b2`): 43 cells (Severson Primary Test cohort)
  * Batch 3 (`b3`): 40 cells (Severson Secondary Test cohort)

---

## 2. Directory Structure

```
trial_03_mit_lfp/
├── data/                      # Ingested CSV files and split definitions
│   ├── README.md              # Staging instructions and schema documentation
│   ├── mit_cycle_life_clean.csv   # Primary clean 124-cell input (drop here)
│   ├── mit_cycle_life_raw.csv     # Raw feature matrix (drop here)
│   ├── mit_splits_70_30_x20.csv   # 20-seed 70/30 stratified splits
│   └── FEATURES.md            # Mathematical feature formulas
├── reports/                   # Benchmark evaluation results & comparative plots
│   └── severson_baseline_replication.md
└── src/                       # Modeling and evaluation scripts
    ├── ingest_and_preprocess.py       # Imputes b1c18 and formats arrays
    ├── reproduce_severson_baselines.py # Exact replication of Severson 2019 Table 1
    ├── train_mit_hybrid_pghqml.py     # Physics-Guided Hybrid Quantum-Classical model
    └── evaluate_mit_models.py         # Comparative benchmark & error analysis
```

---

## 3. Published Benchmarks to Match & Beat

### Severson et al. (Nature Energy, 2019)
Trained strictly on Batch 1 (`train`, 41 cells); evaluated on Batch 2 (`primary_test`, 43 cells) and Batch 3 (`secondary_test`, 40 cells). Target is $\log_{10}(\text{cycle life})$; reported metric is Root Mean Squared Error (RMSE) in **cycles** ($10^{\hat{y}}$):

| Model | Feature Count & Selection | Primary Test RMSE (cycles) | Secondary Test RMSE (cycles) |
|---|---|---|---|
| **Variance Model** | 1 feature: $\log_{10}(\text{Var}(\Delta Q_{100-2}(V)))$ | **138** | **196** |
| **Discharge Model** | 6 features: `dq_min`, `dq_var`, `dq_skew`, `dq_kurt`, `q_cycle2`, `qmax_minus_q2` | **91** | **173** |
| **Full Model** | 9 features: `dq_min`, `dq_var`, `cap_slope_2_100`, `cap_intercept_2_100`, `q_cycle2`, `charge_time_avg`, `temp_integral`, `ir_min`, `ir_diff_100_2` | **118** | **214** |

### Fei et al. (Energy & Environmental Science, 2021) & Anggita & Akrom (2025)
- Protocol: Repeated 70/30 train/test splits (20 random seeds), reporting mean RMSE and MAPE.
- Note: Uses 42 expanded features; provided in `mit_splits_70_30_x20.csv` as secondary reference benchmarks.

---

## 4. Dual-Chemistry Integration with CALCE CS2

Integrating the MIT LFP cohort with the CALCE LCO dataset establishes the **VoltQuant Dual-Chemistry Prognostics Suite**:

| Dimension | CALCE CS2 Pipeline (`CALSC/`) | MIT Pipeline (`MIT/`) |
|---|---|---|
| **Cell Chemistry** | Prismatic LiCoO$_2$ (LCO) / Graphite | Cylindrical LiFePO$_4$ (LFP) / Graphite |
| **Degradation Mode** | Non-linear rollover / sudden voltage cliff | Gradual capacity fade / impedance rise |
| **Discharge Condition** | Continuous 0.5C discharge | Continuous 4.0C high-rate discharge |
| **Prognostic Focus** | Sub-cycle runaway warning & 72s cliff buffer | Cycle-life regression from first 100 cycles |
| **Advance Horizon** | 10–15 min instant buffer / Cycle 50 screening | Lifetime prediction (148–2,237 cycles) at Cycle 100 |
| **Validation Scheme** | 4-Fold Leave-One-Cell-Out (LOCO) | Batch-held-out (41 / 43 / 40) + 70/30 $\times$ 20 |

