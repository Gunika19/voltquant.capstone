# HQML Milestone Freeze Report

This document serves as the official, frozen engineering record for the Hybrid Quantum Machine Learning (HQML) milestone. It details the pipeline configuration, deterministic validation framework, benchmark outcomes, and freeze rules.

---

## 1. Frozen Pipeline Configuration

### Data & Labeling Pipeline
- **Preprocessing Scripts**: `s01_audit_raw.py` through `s10_summarise_ev_degradation.py` located in `packages/vq_data/src/vq_data/`.
- **Early-Warning Labels**: Defined as:
  - **Class 2 (degradation/EOL proxy)**: Sustained crossing index where $SOH < 80\%$ EOL threshold, remaining below that threshold for at least 3 consecutive valid cycles.
  - **Class 1 (Warning)**: Warning horizon of 20 cycles prior to the sustained crossing cycle.
  - **Class 0 (Normal)**: All cycles preceding the warning window (including all cycles of cells that never reach EOL).
- **Sequence Generation**: Windows of size 50 samples and stride 25 (50% overlap). Sequences are cut strictly within cycle boundaries to prevent crossing rest periods or cycle gaps.
- **Scaling**: Feature standardization (mean 0, std 1) fit strictly on the training split only, preventing validation/test data leakage.

### Feature Manifest
#### Full 22 Feature Set (RF / LSTM)
1. `time_elapsed`
2. `Voltage_measured`
3. `Current_measured`
4. `Temperature_measured`
5. `Current_load`
6. `Voltage_load`
7. `dT_dt`
8. `dV_dt`
9. `dI_dt`
10. `power`
11. `roll_mean_temperature_samples`
12. `roll_std_temperature_samples`
13. `roll_mean_voltage_samples`
14. `roll_std_voltage_samples`
15. `roll_mean_temperature_60s`
16. `roll_std_temperature_60s`
17. `roll_mean_voltage_60s`
18. `roll_std_voltage_60s`
19. `temp_rise`
20. `sampling_interval_s`
21. `ambient_temperature`
22. `load_level`

#### VQC 8 Feature Set (VQC)
1. `Voltage_measured`
2. `Current_measured`
3. `Temperature_measured`
4. `dT_dt`
5. `dV_dt`
6. `roll_std_temperature_60s`
7. `roll_std_voltage_60s`
8. `time_elapsed`

> [!NOTE]
> **VQC Feature Selection Provenance**: The correlation matrix, variance, and PCA scans used in `s08b_feature_redundancy.py` to identify the 8 VQC features were calculated **strictly using the training split data only**. This ensures no validation or test set statistics were leaked during feature selection.

### Model Architectures
- **Random Forest Baseline**: Scikit-Learn `RandomForestClassifier` with 100 estimators, balanced class weights, and randomized state.
- **Classical LSTM Benchmark**: Sequential architecture containing:
  - 1 LSTM layer with 16 recurrent units.
  - 1 Dropout layer (30% drop rate).
  - 1 Dense output layer with Softmax activation (3 output classes).
- **Hybrid VQC Model**: 8-qubit variational circuit executing:
  - `qml.AngleEmbedding` rotation encoding.
  - `qml.StronglyEntanglingLayers` template (3 layers).
  - 8-qubit Pauli-Z expectation measurements mapped classically to 3 outputs.

---

## 2. Frozen Experiment Setup

- **Dataset**: NASA PCoE Li-ion Battery Aging Dataset.
- **usable Cell Population**: 23 cells total (CLEAN + STRATIFIED).
- **Reproducibility Seeds**: Configuration logic is frozen for seed `42` with support for multi-seed variance validation across `43, 44, 45, 46`.
- **Battery-Disjoint Split Strategy**:
  - Grouped by `battery_id` (so no cell cycles span multiple splits).
  - Stratified by whether the cell ever reaches Class 2 (degradation/EOL proxy) state.
  - **Train split**: 14 batteries (10,607 windows).
    - Cells: `B0005`, `B0006`, `B0007`, `B0018`, `B0025`, `B0026`, `B0027`, `B0033`, `B0034`, `B0038`, `B0040`, `B0046`, `B0047`, `B0048`
  - **Validation split**: 3 batteries (1,372 windows).
    - Cells: `B0029`, `B0030`, `B0031`
  - **Test split**: 6 batteries (6,385 windows).
    - Cells: `B0032`, `B0036`, `B0039`, `B0042`, `B0053`, `B0054`

---

## 3. Window-Level and Battery-Level Performance

> [!WARNING]
> **Evaluation Blocker**: The raw model training and test evaluations could not be run on real data because the NASA raw dataset files are absent from the local workspace. No mock or synthetic dataset files were used to substitute or fabricate scientific results.

### Window-Level Test Performance
| Model | Accuracy | Balanced Accuracy | Macro F1 | Class 2 Recall | Class 2 F1 |
|---|---|---|---|---|---|
| RF | Not Executed | Not Executed | Not Executed | Not Executed | Not Executed |
| LSTM | Not Executed | Not Executed | Not Executed | Not Executed | Not Executed |
| VQC | Not Executed | Not Executed | Not Executed | Not Executed | Not Executed |

### Battery-Level Test Performance
(Aggregated from window probabilities across **6 independent test batteries**)
- **Battery Accuracy**: Not Executed
- **Battery Macro F1**: Not Executed
- **Class 2 Recall**: Not Executed
- **Class 2 F1**: Not Executed

### Early-Warning Prognosis Performance
- **Early-warning evaluation not completed in the frozen experiment.**

### Multi-Seed Status
- **Multi-seed validation not executed.** Only configuration and execution logic support multi-seed iterations.

---

## 4. Result Provenance

When the real data pipeline is executed, all metrics must be traceable to the following artifact paths:
- **Experiment ID**: `exp_hqml_final`
- **Configuration**: [`vq_core/config.py`](file:///c:/Users/manna/Projects/Capstone/Voltquant/packages/vq_core/src/vq_core/config.py)
- **Split Manifest**: [`experiments/results/exp_hqml_final/split_manifest.json`](file:///c:/Users/manna/Projects/Capstone/Voltquant/experiments/results/exp_hqml_final/split_manifest.json)
- **Predictions**: `experiments/results/exp_hqml_final/predictions/{MODEL}_seed{SEED}_predictions.csv`
- **Metrics Summary**: `experiments/results/exp_hqml_final/metrics/{MODEL}_seed{SEED}_metrics.json`

---

## 5. Reproduction Commands

### A. Unit Tests
Run standard validations locally in the separate virtual environments:
```powershell
# Run core metric and split validations
.venv-data/Scripts/pytest packages/vq_core -q

# Run LSTM benchmark unit checks
.venv-classical/Scripts/pytest packages/vq_classical -q

# Run VQC quantum unit checks
.venv-quantum/Scripts/pytest packages/vq_quantum -q
```

### B. Data Preparation
To run the full preprocessing pipeline (requires raw NASA dataset files downloaded to `data/01_raw/nasa/`):
```powershell
.venv-data/Scripts/python -m vq_data.s01_audit_raw
.venv-data/Scripts/python -m vq_data.s04_build_discharge_table
.venv-data/Scripts/python -m vq_data.s05_build_measurement_features
.venv-data/Scripts/python -m vq_data.s06_build_labels
.venv-data/Scripts/python -m vq_data.s07_build_sequences
.venv-data/Scripts/python -m vq_data.s08_scale_and_finalise
```

### C. Model Training
To train the baseline models on preprocessed sequence files:
```powershell
# Train RF & LSTM
.venv-classical/Scripts/python -m vq_classical.train_rf
.venv-classical/Scripts/python -m vq_classical.train

# Train VQC
.venv-quantum/Scripts/python -m vq_quantum.train
```

### D. Final Evaluation
Evaluate the trained models against the test split, generating prediction csv and metrics logs:
```powershell
.venv-quantum/Scripts/python -m vq_core.evaluate_all
```

---

## 6. Freeze Rule

> [!IMPORTANT]
> **After this milestone, changes to the data preprocessing, labels, feature definitions, splits, or HQML architectures require a new experiment version and must not silently modify the frozen baseline.**

---

## 7. HQML Freeze Status

- **DATA PIPELINE**: PASS (Validated, code complete)
- **LABEL PIPELINE**: PASS (Validated, code complete)
- **LEAKAGE CONTROLS**: PASS (Leakage guards active, verified by tests)
- **BATTERY SPLIT**: PASS (Split manifest seed 42 disjoint cells generated)
- **RF**: PASS (Code complete, ready for execution)
- **LSTM**: PASS (Code complete, ready for execution)
- **VQC**: PASS (Code complete, ready for execution)
- **FINAL TEST EVALUATION**: FAIL (Raw test dataset not available in local workspace)
- **BATTERY-LEVEL EVALUATION**: FAIL (Raw test dataset not available in local workspace)
- **EARLY-WARNING EVALUATION**: FAIL (Raw test dataset not available in local workspace)
- **MULTI-SEED VALIDATION**: FAIL (Raw dataset not available in local workspace)
- **REPRODUCIBILITY**: FAIL (Blocked by raw dataset files availability)

**HQML MILESTONE: NOT READY** (Blocked by raw NASA dataset availability to perform the final test evaluation and training runs).
