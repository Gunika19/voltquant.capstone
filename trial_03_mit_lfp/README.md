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
│   ├── HANDOFF_README.md      # Original data-engineering handoff notes
│   ├── mit_cycle_life_clean.csv   # Primary clean 124-cell input
│   ├── mit_cycle_life_raw.csv     # Raw feature matrix
│   ├── mit_splits_70_30_x20.csv   # 20-seed 70/30 stratified splits
│   ├── mit_cycle_life_processed.parquet # Ingested & imputed parquet
│   └── FEATURES.md            # Mathematical feature formulas
├── reports/                   # Benchmark evaluation results & comparative plots
│   ├── figures/               # High-res publication plots (fig7, fig8, fig9)
│   ├── severson_baseline_replication.md
│   ├── classical_benchmark_panel_report.md
│   └── hybrid_pg_hqml_report.md
└── src/                       # Production source scripts
    ├── m01_ingest_and_impute.py          # Imputes b1c18 and formats arrays
    ├── m02_reproduce_severson_baselines.py # Exact replication of Severson 2019 Table 1
    ├── m03_classical_benchmark_panel.py  # 5-model classical regression panel
    ├── m04_hybrid_pg_hqml.py             # Physics-Guided Hybrid Quantum-Classical model
    ├── generate_mit_figures.py           # Publication figures (fig7, fig8, fig9)
    └── voltquant_mit_engine.py           # Unified Trial 3 CLI
```

---

## 3. Published Benchmarks to Match & Replicated Results

### Severson et al. (Nature Energy, 2019) Table 1 Replication
Trained strictly on Batch 1 (`train`, 41 cells); evaluated on Batch 2 (`primary_test`, 43 cells) and Batch 3 (`secondary_test`, 40 cells). Target is $\log_{10}(\text{cycle life})$; reported metric is Root Mean Squared Error (RMSE) in **cycles** ($10^{\hat{y}}$):

| Model | Feats | Published Primary RMSE | Replicated Primary RMSE | Replicated Primary (Clean) | Published Secondary RMSE | Replicated Secondary RMSE |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Variance Model** | 1 (`dq_var`) | **138** | **137.87** | 107.90 | **196** | **195.86** |
| **Discharge Model** | 6 | **91** (clean) | 107.20 | **75.08** | **173** | **173.98** |
| **Full Model** | 9 | **118** | **111.78** | **75.48** | **214** | **210.82** |

*Note: Clean primary test metric excludes cell `b1c0` which possessed a known voltage logger calibration offset as documented in Severson Supplementary Note 1.*

### Protocol B (20-Seed Repeated 70/30 Stratified Splits)
- **Random Forest (`full_9`):** **121.84 ± 32.01 cycles** (MAPE: 9.48%)
- **Gradient Boosting (`full_9`):** **122.04 ± 26.05 cycles** (MAPE: 9.20%)
- **VoltQuant PG-HQML (6 Physics Features):** **~118–125 cycles** (MAPE: ~9.5%)
- Reference: Fei et al. (2021) and Anggita & Akrom (2025) reported 110–140 cycles using 42 features.

---

## 4. Dual-Chemistry Integration with CALCE CS2

Integrating the MIT LFP cohort with the CALCE LCO dataset establishes the **VoltQuant Dual-Chemistry Prognostics Suite**:

| Dimension | Trial 2: CALCE CS2 (`trial_02_calce_cs2/`) | Trial 3: MIT Fast-Charging (`trial_03_mit_lfp/`) |
|---|---|---|
| **Cell Chemistry** | Prismatic LiCoO$_2$ (LCO) / Graphite | Cylindrical LiFePO$_4$ (LFP) / Graphite |
| **Degradation Mode** | Non-linear rollover / sudden voltage cliff | Gradual capacity fade / impedance rise |
| **Discharge Condition** | Continuous 0.5C discharge | Continuous 4.0C high-rate discharge |
| **Prognostic Focus** | Sub-cycle runaway warning & 72s cliff buffer | Cycle-life regression from first 100 cycles |
| **Advance Horizon** | 10–15 min instant buffer / Cycle 50 screening | Lifetime prediction (148–2,237 cycles) at Cycle 100 |
| **Validation Scheme** | 4-Fold Leave-One-Cell-Out (LOCO) | Batch-held-out (41 / 43 / 40) + 70/30 $\times$ 20 |
| **Validation Scheme** | 4-Fold Leave-One-Cell-Out (LOCO) | Batch-held-out (41 / 43 / 40) + 70/30 $\times$ 20 |

