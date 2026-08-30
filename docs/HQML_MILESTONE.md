# HQML Milestone Freeze Report

This document serves as the official, frozen engineering record for the Hybrid Quantum Machine Learning (HQML) milestone. It details the pipeline configuration, deterministic validation framework, benchmark outcomes, and freeze rules.

---

## 1. Frozen Pipeline Configuration

### Data & Labeling Pipeline
- **Preprocessing Scripts**: `s01_audit_raw.py` through `s10_summarise_ev_degradation.py` located in `packages/vq_data/src/vq_data/`.
- **Early-Warning Labels**: Defined case-insensitively using:
  - **Class 2 (Runaway)**: Sustained crossing index where $SOH < 80\%$ EOL threshold, remaining below that threshold for at least 3 consecutive valid cycles.
  - **Class 1 (Warning)**: Warning horizon of 20 cycles prior to the sustained crossing cycle.
  - **Class 0 (Normal)**: All cycles preceding the warning window (including all cycles of cells that never reach EOL).
- **Sequence Generation**: Windows of size 50 samples and stride 25 (50% overlap). Sequences are cut strictly within cycle boundaries to prevent crossing rest periods or cycle gaps.
- **Scaling**: Feature standardization (mean 0, std 1) fit strictly on the training split only, preventing validation/test data leakage.

### Feature Subsets
- **Full Features (LSTM / RF)**: 22 auto-detected numeric features containing raw measurements, first derivatives, and rolling statistical variations.
- **VQC Features (VQC)**: 8 designated features hand-selected based on correlation and PCA analysis:
  - `Voltage_measured`, `Current_measured`, `Temperature_measured`, `dT_dt`, `dV_dt`, `roll_std_temperature_60s`, `roll_std_voltage_60s`, `time_elapsed`.

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
- **Reproducibility Seeds**: Baseline seed `42` with support for multi-seed variance validation across `43, 44, 45, 46`.
- **Battery-Disjoint Split Strategy**:
  - Grouped by `battery_id` (so no cell cycles span multiple splits).
  - Stratified by whether the cell ever reaches runaway.
  - **Train split**: 14 batteries (10,607 windows).
  - **Validation split**: 3 batteries (1,372 windows).
  - **Test split**: 6 batteries (6,385 windows).

---

## 3. Reference Results

Authoritative experiment outputs and logs are persisted under:
`experiments/results/exp_hqml_final/`

Which contains:
- `split_manifest.json`: seed, cell splits, window sizes, and class balance counts.
- `predictions/`: raw prediction probability logs (`predictions.csv`) for window-level analysis.
- `metrics/`: window-level, battery-level aggregated, and early-warning prognostic metric JSONs.

---

## 4. Known Limitations

1. **Proxy Target Application**: The dataset does not contain physical thermal-runaway events. The 80% SOH EOL threshold acts as an experimental prognostic proxy for degradation-induced critical safety events.
2. **Sampling Rate Variations**: Sampling intervals are not uniform across cycles (varying from 9.3s to 20.5s). Downstream models must condition on `sampling_interval_s`.
3. **EOL Metric Sensitivity**: The EOL crossing cycle is highly sensitive to the capacity reference estimator. The pipeline operates strictly on a top-5 highest valid capacity median reference.

---

## 5. Reproduction Commands

To verify the test suite, run the following commands in their respective virtual environments:

```powershell
# Run core validation and metric tests
.venv-data/Scripts/pytest packages/vq_core -q

# Run classical benchmark tests
.venv-classical/Scripts/pytest packages/vq_classical -q

# Run quantum VQC tests
.venv-quantum/Scripts/pytest packages/vq_quantum -q
```

To run training/evaluation, ensure raw NASA datasets are placed in `data/` and execute:
```powershell
.venv-classical/Scripts/python -m vq_classical.train
.venv-quantum/Scripts/python -m vq_quantum.train
```

---

## 6. Freeze Rule

> [!IMPORTANT]
> **After this milestone, changes to the data preprocessing, labels, feature definitions, splits, or HQML architectures require a new experiment version and must not silently modify the frozen baseline.**
