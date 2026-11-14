# Trial 1: NASA PCoE Battery Prognostics (Exploratory Baseline)

This directory contains the initial exploratory codebase, packages, virtual environment specifications, and experiment notebooks developed on the **NASA Ames Prognostics Center of Excellence (PCoE)** Li-ion battery dataset (`B0005`, `B0006`, `B0007`, `B0018`).

---

## 1. Scope & Technical Architecture

* **Dataset:** 4 cylindrical 18650 LiCoO$_2$ cells cycled at room temperature (24°C) under repeated 2A discharge cycles to 2.7V, 2.5V, 2.2V cutoffs.
* **Target Objective:** Predict Remaining Useful Life (RUL) and State of Health (SOH) degradation using classical and quantum algorithms.
* **Modular Packages:**
  * `packages/vq_core`: Configuration, base schema, metric calculations, and common logging utilities.
  * `packages/vq_data`: Data loaders and feature engineering pipelines for NASA PCoE `.mat` files.
  * `packages/vq_classical`: Classical machine learning models (Random Forest, LightGBM, XGBoost, and classical baseline MLP).
  * `packages/vq_quantum`: Standalone PennyLane Variational Quantum Circuit (VQC) architectures.

---

## 2. Comprehensive Scientific Audit & Failure Mode Analysis

During formal project auditing, this pipeline was evaluated against rigorous production battery management requirements. The audit revealed three fatal architectural limitations that necessitated progressing to **Trial 2 (CALCE CS2)**:

1. **Extreme Sample Starvation:**
   - The NASA cohort contains only **4 cells** in total.
   - Traditional train/test splits (3 cells train, 1 cell test) possess near-zero statistical power and cannot capture cell-to-cell manufacturing variance.

2. **Severe Temporal Label Leakage in Exploratory Feature Pipelines:**
   - Up to **86% label leakage** was identified in exploratory rolling-window features, where future capacity metrics or whole-lifetime normalization statistics leaked into historical training cycles.
   - When evaluated under strict causal validation gates, nominal model accuracy collapsed significantly.

3. **Catastrophic Out-of-Distribution Collapse (0.00% Warning Recall):**
   - The standalone PennyLane VQC model demonstrated severe gradient vanishing and barren plateaus when scaling to unconstrained parameter spaces.
   - When tested on held-out out-of-distribution cells undergoing non-linear knee degradation, the standalone VQC achieved **0.00% true warning recall** on critical warning cycles.

---

## 3. Directory Structure

```
trial_01_nasa_pcoe/
├── archive/                   # Initial exploration prototypes & v1 checkpoints
├── envs/                      # Dependency lockfiles (classical.txt, quantum.txt, data.txt)
├── experiments/               # Experiment tracking notebooks and tabular outputs
│   └── results/               # Classical and quantum benchmark results
├── packages/                  # Modular Python packages
│   ├── vq_classical/          # LightGBM, XGBoost, RF implementations
│   ├── vq_core/               # Schema, metric definitions, base classes
│   ├── vq_data/               # NASA PCoE feature extractors
│   └── vq_quantum/            # PennyLane VQC standalone models
├── reports/                   # Empirical degradation summary reports and figures
└── README.md                  # This audit and architecture document
```

---

## 4. Transition to Subsequent Trials

To overcome these fundamental limitations:
- **Trial 2 (`trial_02_calce_cs2`):** Transitioned to the CALCE CS2 continuous 0.5C dataset (900+ cycles/cell), implemented a strict zero-leakage causal validation gate, and pioneered the **Physics-Guided Hybrid Quantum-Classical (PG-HQML)** architecture with dual-horizon early warning buffers.
- **Trial 3 (`trial_03_mit_lfp`):** Scaled to commercial fleet dimensions on 124 fast-charging cells across 3 production batches from the MIT/Stanford/Toyota dataset.
