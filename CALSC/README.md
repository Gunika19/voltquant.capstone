# CALSC: Physics-Guided Hybrid Quantum-Classical Prognostics on CALCE CS2

This folder contains the complete, out-of-distribution battery prognostics and early runaway warning software pipeline developed on the **University of Maryland CALCE CS2 cycling dataset** (LiCoO$_2$/graphite prismatic cells, continuous 0.5C discharge, 900+ cycles per cell).

---

## 1. Pipeline Architecture & Directory Structure

```
CALSC/
├── data/                      # Local data directory (raw zips excluded from git tracking)
│   └── README.md              # Instructions for downloading CALCE data
├── reports/                   # Empirical engineering benchmark reports
│   ├── figures/               # High-resolution publication figures (fig1 to fig6)
│   ├── k01_audit_report.md
│   ├── k06_validation_gate_report.md
│   ├── k07_classical_panel_report.md
│   ├── k08_vqc_report.md
│   ├── k13_hybrid_pg_hqml_report.md
│   ├── k14_ultra_early_timing_report.md
│   ├── k16_robust_warning_system_report.md
│   └── ...
└── src/                       # Production source scripts
    ├── k01_acquire_and_audit.py                  # Download and inspect raw logs
    ├── k02_build_cycle_table.py                  # Parse discharge cycles & SOH
    ├── k03_build_features.py                     # Extract 66 rolling window features
    ├── k04_k05_build_sequences.py                # Create 4-fold LOCO splits
    ├── k06_validate_pipeline.py                  # Zero label leakage gate
    ├── k07_classical_panel.py                    # Classical ML benchmark (XGBoost, RF, etc.)
    ├── k08_vqc_model.py                          # Standalone quantum benchmark (VQC)
    ├── k12_early_timing_features.py              # 8 Physics-guided non-destructive features
    ├── k13_hybrid_pg_hqml.py                     # PyTorch + PennyLane PG-HQML model
    ├── k14_ultra_early_timing_pipeline.py        # Cycle 50 & 100 early lifetime screening
    ├── k15_within_cycle_early_warning.py         # Sub-cycle window prediction
    ├── k16_robust_warning_system.py              # K=3 persistence filter & AES-128-CMAC
    ├── k17_early_timing_in_minutes.py            # Quantitative time in minutes/hours
    ├── k18_instant_warning_and_literature_benchmarks.py  # 10-15 min advance buffer & literature
    ├── generate_paper_figures.py                 # Generates publication figures fig1 - fig5
    ├── generate_fig6.py                          # Generates runaway cliff figure fig6
    └── voltquant_engine.py                       # Unified production CLI
```

---

## 2. Key Scientific Findings & Benchmark Highlights

* **Out-of-Distribution Leave-One-Cell-Out (LOCO) Generalization:**
  * **LOCO Warning Recall:** **78.14%** (Classical XGBoost: 60.23% | Standalone VQC: 32.14% | NASA Baseline: 0.00%).
  * **Balanced Accuracy:** **86.63%** across all 4 held-out cells (`CS2_35`, `CS2_36`, `CS2_37`, `CS2_38`).
  * **Macro F1:** **83.28%** with **0 / 4 collapsed folds**.
* **Dual-Horizon Early Timing Framework:**
  * **Sub-Cycle Instant Runaway Warning:** Detects degradation precursors within the first 10–15 minutes of discharge ($t \le 600\text{–}900\text{ s}$), delivering an **exact 10.0 to 15.0 minute advance safety buffer** before the catastrophic 72-second voltage collapse cliff ($3.0\text{ V} \to 2.75\text{ V}$).
  * **Ultra-Early Lifecycle Screening:** At **Cycle 50** (~5.6% of lifespan), predicts End-of-Life failure (80% SOH) with an MAE of **38.64 cycles**, providing **27,092 operational minutes (451.5 hours / ~20,300 km)** of advance maintenance notice.
* **Zero-Thermocouple Observability:** Operates strictly on terminal electrical readings ($V, I, t$). Zero internal cell-level thermocouples required.
* **Zero False Alarms:** A $K=3$ temporal persistence filter suppresses capacity regeneration anomalies, achieving **0.00% False Alarms** across 841 healthy operational cycles.

---

## 3. Quickstart & Execution

### Prerequisites
Install dependencies in your Python 3.10+ environment:
```bash
pip install torch pennylane xgboost lightgbm scikit-learn pandas numpy matplotlib cryptography
```

### Running the Unified CLI
```bash
# 1. Run the runaway cliff speed and advance buffer analysis:
python src/voltquant_engine.py runaway

# 2. Simulate real-time discharge monitoring with automated alert transitions:
python src/voltquant_engine.py stream --cell CS2_35 --cycle 700

# 3. Perform ultra-early lifetime screening at Cycle 50:
python src/voltquant_engine.py screen --cell CS2_35 --cycle 50

# 4. Display the peer-reviewed literature matrix (2023-2025):
python src/voltquant_engine.py literature

# 5. Display the full LOCO evaluation metrics:
python src/voltquant_engine.py evaluate
```

### Running Model Training
```bash
# Train and evaluate the PG-HQML model across all 4 LOCO folds:
python src/k13_hybrid_pg_hqml.py

# Or run a single fold test:
python src/k13_hybrid_pg_hqml.py --fold 0
```
