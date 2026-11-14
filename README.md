# VoltQuant: Physics-Guided Hybrid Quantum-Classical Battery Prognostics

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/Quantum-PennyLane-brightgreen.svg)](https://pennylane.ai/)
[![Validation](https://img.shields.io/badge/Leakage%20Gate-Verified%200%25-success.svg)]()
[![LOCO Recall](https://img.shields.io/badge/LOCO%20Warning%20Recall-78.14%25-blueviolet.svg)]()

VoltQuant is a physics-guided, hybrid quantum-classical machine learning (PG-HQML) prognostics suite designed for commercial lithium-ion energy storage systems and electric vehicle battery packs. VoltQuant integrates physics-guided degradation signatures, quantum variational circuits (VQC), and cryptographic telemetry authentication (AES-128-CMAC) to detect battery failure precursors well in advance of catastrophic voltage collapse.

---

## Repository Structure (Organized by Experimental Trials)

This repository is strictly partitioned into three distinct experimental trials representing the scientific evolution of the project:

```
voltquant.capstone/
├── trial_01_nasa_pcoe/        # TRIAL 1: NASA PCoE Baseline (18650 LiCoO2)
│   ├── packages/              # Modular Python packages (vq_classical, vq_quantum, vq_data, vq_core)
│   ├── experiments/           # Baseline training experiments and metric JSONs
│   ├── envs/                  # Environment lockfiles (classical.txt, quantum.txt, data.txt)
│   ├── archive/               # Historical v1 Kedro exploration pipeline
│   ├── reports/               # Exploratory degradation reports & figures
│   └── README.md              # Trial 1 overview, audit findings, and transition rationale
│
├── trial_02_calce_cs2/        # TRIAL 2: CALCE CS2 Production Engine (Prismatic LCO)
│   ├── data/                  # Data directory (raw 374MB zips excluded via .gitignore)
│   ├── reports/               # Formal benchmark reports and publication figures (fig1-fig6)
│   ├── src/                   # 18 production Python scripts & unified voltquant_engine.py CLI
│   └── README.md              # Trial 2 PG-HQML architecture, LOCO validation, and CLI guide
│
├── trial_03_mit_lfp/          # TRIAL 3: MIT / Stanford Commercial Cohort (A123 LFP)
│   ├── data/                  # Ingestion schema & staging instructions for 124-cell CSVs
│   ├── reports/               # Baseline replication reports against Severson 2019 Table 1
│   ├── src/                   # Ingestion, imputation, and early cycle-life prediction scripts
│   └── README.md              # Trial 3 objectives, feature formulas, and benchmark metrics
│
├── .github/                   # CI/CD and automation workflows
├── .gitignore                 # Enforces exclusion of heavy data (374MB) & scratch files
└── README.md                  # Master repository index (this file)
```

---

## Evolutionary Summary of the 3 Trials

| Dimension | Trial 1: NASA PCoE (`trial_01_nasa_pcoe`) | Trial 2: CALCE CS2 (`trial_02_calce_cs2`) | Trial 3: MIT / Stanford (`trial_03_mit_lfp`) |
|---|---|---|---|
| **Cell Format & Chemistry** | Cylindrical 18650 LiCoO$_2$ | Prismatic LiCoO$_2$ (CS2) | Cylindrical 18650 $\text{LiFePO}_4$ (A123) |
| **Testing Regime** | Low-rate cycling (2.0A) | Continuous 0.5C discharge (900+ cyc) | 72 Fast-charging protocols (4.0C disch.) |
| **Cohort Size** | 4 cells total | 4 cells (deep cycling, 3,600+ cyc) | 124 commercial cells (3 batches) |
| **Primary Task** | Exploratory RUL / SOH classification | Sub-cycle early warning & screening | Cycle-life prediction at Cycle 100 |
| **Validation Scheme** | Random / 3 train, 1 test | 4-Fold Leave-One-Cell-Out (LOCO) | Batch-held-out (41 / 43 / 40) + 70/30 |
| **Key Benchmark** | Baseline / 0.00% warning recall | **78.14% LOCO Recall**, 86.63% Bal Acc | Severson 2019 (138/91/118 cyc RMSE) |
| **Production Role** | Historical negative control | Active Production Engine | Active Multi-Batch Benchmark |

---

## Quickstart & Execution

### Installation
```bash
git clone https://github.com/Gunika19/voltquant.capstone.git
cd voltquant.capstone
pip install torch pennylane xgboost lightgbm scikit-learn pandas numpy matplotlib cryptography
```

### Running Trial 2 (CALCE CS2 Production Engine)
All Trial 2 benchmarks and analyses are accessible via `trial_02_calce_cs2/src/voltquant_engine.py`:
```bash
# 1. Run runaway cliff speed and advance buffer analysis:
python trial_02_calce_cs2/src/voltquant_engine.py runaway

# 2. Simulate real-time discharge telemetry stream:
python trial_02_calce_cs2/src/voltquant_engine.py stream --cell CS2_35 --cycle 700

# 3. Perform ultra-early lifetime screening at Cycle 50:
python trial_02_calce_cs2/src/voltquant_engine.py screen --cell CS2_35 --cycle 50

# 4. Display 2023-2025 comparative literature matrix:
python trial_02_calce_cs2/src/voltquant_engine.py literature

# 5. Train the full 4-fold Leave-One-Cell-Out PG-HQML model:
python trial_02_calce_cs2/src/k13_hybrid_pg_hqml.py
```

### Running Trial 3 (MIT Fast-Charging Cohort)
Place the clean dataset `mit_cycle_life_clean.csv` in `trial_03_mit_lfp/data/` and execute:
```bash
python trial_03_mit_lfp/src/reproduce_severson_baselines.py
```

---

## License & Citation
Developed for the Capstone Project. For full technical details and theoretical proofs, see the documentation in `trial_02_calce_cs2/reports/` and `trial_01_nasa_pcoe/README.md`.
