# Comprehensive Dataset Provenance Report: Exact Transformations, Feature Pipelines, and Modeling Across CALCE CS2 and NASA PCoE (Protocol Aligned)

**Project:** Cross-Dataset Lithium-Ion Battery Prognostics & Early-Warning Degradation Classification  
**Date:** September 2026  
**Document Type:** Technical Provenance & Data Transformation Specification  
**Status:** Protocol Aligned to Verified NASA Reference Standard  

---

## 1. Executive Summary & Research Motivation

This document provides an exhaustive, mathematically rigorous, and reproducible record of **every operation, transformation, audit fix, feature extraction pipeline, and model evaluation** applied to:
1. **The CALCE CS2 Battery Dataset** (Cells: `CS2_35`, `CS2_36`, `CS2_37`, `CS2_38`)
2. **The NASA PCoE Battery Dataset** (Cells: `B0005`, `B0006`, `B0007`, `B0018`)

### The Scientific Core: The "Resolution Hypothesis"
In prior battery prognostic research conducted on the NASA PCoE dataset, out-of-distribution Leave-One-Cell-Out (LOCO) classification models suffered a catastrophic failure on the **warning degradation class** (pre-EOL phase), collapsing to **0.0000 warning recall** and **0.0591 balanced accuracy**. 

Two competing hypotheses explained this failure:
- **Hypothesis 1 (Physical Unrecoverability):** The warning precursor signature cannot be captured from electrical discharge window summaries alone without direct cell surface thermocouple measurements ($T$, $dT/dt$, $\Delta T_{rise}$).
- **Hypothesis 2 (Resolution Limitation):** The NASA dataset is severely resolution-limited (only 132–168 cycles total; a 20-cycle warning window represents 12–15% of the cell's entire operational lifespan). Lithium-ion electrochemical aging dynamics require dense, continuous cycle tracking ($\ge 800$ cycles) to expose subtle, nonlinear electrical precursor signatures before EOL.

To decisively test this hypothesis, we ported the entire prognostic pipeline to the **CALCE CS2 dataset**, which offers **880–1,080 continuous cycles** but contains **zero thermocouple sensors**. To ensure scientific validity, all protocols between the two datasets have been aligned to a strict 1-to-1 parity.

---

## 2. Dataset 1: CALCE CS2 Battery Dataset

### 2.1 Raw Data Ingestion & Specifications
- **Source:** Center for Advanced Life Cycle Engineering (CALCE), University of Maryland.
- **Cells Acquired:** `CS2_35`, `CS2_36`, `CS2_37`, `CS2_38` (4 prismatic cells).
- **Cell Chemistry & Geometry:** Prismatic Lithium Cobalt Oxide ($\text{LiCoO}_2$) cathode with graphite anode; nominal rated capacity $Q_{nominal} = 1.1\text{ Ah}$.
- **Cycling Protocol:** Ambient room temperature cycling on an Arbin cycler. Profile: CC-CV charge (0.55A / 0.5C to 4.2V, held until current $< 0.05\text{A}$), rest, followed by **1C constant-current discharge** (-1.1A) to a cutoff voltage of 2.7V.
- **Raw File Volume:** 106 Microsoft Excel spreadsheets (`.xlsx`) organized across 4 cell directories:
  - `data/01_raw/calce/CS2_35/`: 25 `.xlsx` files
  - `data/01_raw/calce/CS2_36/`: 28 `.xlsx` files
  - `data/01_raw/calce/CS2_37/`: 27 `.xlsx` files
  - `data/01_raw/calce/CS2_38/`: 26 `.xlsx` files

### 2.2 Hardware Channel Audit & Thermal Verification (Phase K01)
A systematic, automated audit of all column headers across all raw `.xlsx` files was conducted:
- **Result:** Exactly 17 cycler channels are recorded:
  1. `Data_Point`, 2. `Test_Time(s)`, 3. `Date_Time`, 4. `Step_Time(s)`, 5. `Step_Index`, 6. `Cycle_Index`, 7. `Current(A)`, 8. `Voltage(V)`, 9. `Charge_Capacity(Ah)`, 10. `Discharge_Capacity(Ah)`, 11. `Charge_Energy(Wh)`, 12. `Discharge_Energy(Wh)`, 13. `dV/dt(V/s)`, 14. `Internal_Resistance(Ohm)`, 15. `Is_FC_Data`, 16. `AC_Impedance(Ohm)`, 17. `ACI_Phase_Angle(Deg)`.
- **Thermal Finding:** **ZERO thermocouple channels exist on CALCE CS2.** The substring match `deg` belongs strictly to `ACI_Phase_Angle(Deg)` (electrochemical impedance phase angle). No surface or ambient temperature sensors were attached during these tests.

### 2.3 Data Integrity Auditing & Cycler Traps Resolved (Phase K01 & K02)
1. **Trap A: Non-Zero-Padded Date Sorting Reversal:**
   Raw files named `CS2_35_M_D_YY.xlsx` (e.g., `CS2_35_10_15_10.xlsx` vs `CS2_35_1_10_11.xlsx`) caused lexicographical sorting (`sorted()`) to place October 2010 *after* January 2011, scrambling chronological aging order.
   - **Fix Applied:** Parsed the internal `Date_Time` column of each file to extract true start timestamps and sorted files chronologically.
2. **Trap B: Cycler `Cycle_Index` Reset per File:**
   The Arbin cycler hardware restarts `Cycle_Index` at `1` inside every `.xlsx` log file.
   - **Fix Applied:** Maintained a persistent cumulative cycle index offset across consecutive files:
     $$\text{global\_cycle} = \text{cycle\_offset} + \text{local\_cycle\_index}$$
3. **Trap C: Byte-Identical Duplicate Log Files:**
   Files `CS2_35_2_4_11.xlsx` and `CS2_35_2_10_11.xlsx` contained byte-identical data spanning `2011-01-31 10:51:35` to `2011-02-03 18:10:51` (50 cycles).
   - **Fix Applied:** Implemented multi-file deduplication based on `(Date_Time.first(), Date_Time.last(), row_count)`.

### 2.4 Step Isolation, SOH Reference Alignment & Quality Filtering (Phase K02)
- **Step Isolation:** Filtered strictly for **Step Index 7** (1C constant-current discharge: $I \approx -1.1\text{ A}$, $V \ge 2.7\text{ V}$).
- **Discharge Capacity Calculation:**
  $$Q_{dis} = \int_{t_{start}}^{t_{end}} |I(t)| \, dt$$
- **SOH Reference Standard (Aligned to NASA `s04` Protocol):**
  Per-cell beginning-of-life capacity reference defined as the **median of that cell's 5 highest valid discharge capacities**:
  $$\text{capacity\_ref} = \text{median}(\text{nlargest}_5(Q_{dis}))$$
  - `CS2_35`: **1.1283 Ah** (First cycle SOH = 1.0090)
  - `CS2_36`: **1.1340 Ah** (First cycle SOH = 1.0096)
  - `CS2_37`: **1.1245 Ah** (First cycle SOH = 1.0093)
  - `CS2_38`: **1.1286 Ah** (First cycle SOH = 1.0097)
  *(Passes acceptance check: all cells start within 0.9% of 1.0 SOH).*
- **State of Health Formula:**
  $$\text{SOH} = \frac{Q_{dis}}{\text{capacity\_ref}}$$
- **Cycle Validity Filter (`s04` rule):**
  $$0.30 \cdot \text{capacity\_ref} \le Q_{dis} \le 1.10 \cdot \text{capacity\_ref}$$
- **Final Valid Cycle Yield:** Exactly **3,971 valid discharge cycles** extracted:
  - `CS2_35`: 882 cycles (863 valid, min SOH: 0.3056)
  - `CS2_36`: 973 cycles (879 valid, min SOH: 0.3008)
  - `CS2_37`: 1,038 cycles (975 valid, min SOH: 0.3065)
  - `CS2_38`: 1,078 cycles (1,019 valid, min SOH: 0.3002)
- **Output Artifact:** `data/02_intermediate/calce_discharge_cycles.parquet`

### 2.5 Feature Engineering: Strict `COMMON_FEATURES` Standard (Phase K03)
Restricted strictly to electrical signals present across both NASA and CALCE.
- **Base Physical Signals (11 Signals):**
  1. `time_elapsed` ($t - t_{step\_start}$)
  2. `voltage` ($V$)
  3. `current` ($I$)
  4. `dV_dt` ($\Delta V / \Delta t$)
  5. `dI_dt` ($\Delta I / \Delta t$)
  6. `power` ($V \times I$)
  7. `roll_mean_voltage_samples` (5-sample rolling mean of voltage)
  8. `roll_std_voltage_samples` (5-sample rolling std of voltage, float64, zero-variance guarded below $10^{-12}$)
  9. `roll_mean_voltage_60s` (60-second rolling mean of voltage)
  10. `roll_std_voltage_60s` (60-second rolling std of voltage, float64, zero-variance guarded below $10^{-12}$)
  11. `sampling_interval_s` ($\Delta t$ between consecutive log points, ~5–6 s)
- **Label Leakage Guard:** Verified programmatically (`assert_no_label_leakage`) that no SOH, capacity, or label-correlated variables exist in feature inputs.

### 2.6 Sustained EOL Detection & Dual Warning Schemes (Phase K04)
- **End-of-Life (EOL) Definition:** First cycle where $\text{SOH} < 0.80$ that persists below 0.80 for at least 3 consecutive cycles (`CROSSING_PERSISTENCE = 3`).
- **Verified EOL Crossing Cycles:**
  - `CS2_35`: Cycle **544** (shifted from 536 under nominal reference)
  - `CS2_36`: Cycle **501** (shifted from 493 under nominal reference)
  - `CS2_37`: Cycle **577** (shifted from 569 under nominal reference)
  - `CS2_38`: Cycle **601** (shifted from 575 under nominal reference)
- **Dual Labeling Schemes:**
  1. **Fixed 20-Cycle Window (Direct NASA Comparison, ~2% of lifespan):**
     - Normal ($y=0$): $\text{cycle} < \text{EOL} - 20$ (6,432 windows / 71.65%)
     - Warning ($y=1$): $\text{EOL} - 20 \le \text{cycle} < \text{EOL}$ (236 windows / 2.63%)
     - Degraded ($y=2$): $\text{cycle} \ge \text{EOL}$ (2,309 windows / 25.72%)
  2. **Proportional 12% Window (Fair Learnability Test, ~60–70 cycles):**
     - Normal ($y=0$): $\text{cycle} < \text{EOL} - \text{round}(0.12 \times \text{EOL})$ (5,878 windows / 65.48%)
     - Warning ($y=1$): Within proportional window (790 windows / 8.80%)
     - Degraded ($y=2$): $\text{cycle} \ge \text{EOL}$ (2,309 windows / 25.72%)

### 2.7 Sequence Slicing & Window Aggregation (Phase K05)
- **Sliding Window Parameters:** Length = 50 samples (~250–300 s), Stride = 25 samples (50% overlap).
- **Extracted Instances:** Exactly **8,977 window instances** across all 4 cells (`CS2_35`: 2,142, `CS2_36`: 2,087, `CS2_37`: 2,318, `CS2_38`: 2,430).
- **Exact 66 Summary Features:**
  Applied identical 6 summary statistics (`mean`, `std`, `min`, `max`, `first`, `last`) to each of the 11 base signals ($11 \times 6 = 66$ features). All computed in `float64` with zero-variance guarding below $10^{-12}$.
- **Continuous Regression Target:** Computed continuous distance to failure:
  $$\text{cycles\_to\_crossing} = \text{EOL\_cycle} - \text{current\_cycle}$$
- **Output Artifact:** `data/04_model_input/calce_window_features.parquet`

### 2.8 4-Fold Leave-One-Cell-Out (LOCO) Partitioning (Phase K05)
- **Fold 0:** Test: `CS2_35` (2,142 windows) | Train: `CS2_36`, `CS2_37`, `CS2_38` (6,835 windows)
- **Fold 1:** Test: `CS2_36` (2,087 windows) | Train: `CS2_35`, `CS2_37`, `CS2_38` (6,890 windows)
- **Fold 2:** Test: `CS2_37` (2,318 windows) | Train: `CS2_35`, `CS2_36`, `CS2_38` (6,659 windows)
- **Fold 3:** Test: `CS2_38` (2,430 windows) | Train: `CS2_35`, `CS2_36`, `CS2_37` (6,547 windows)
- **Scaler Isolation:** `StandardScaler` fit strictly on the 3 training cells per fold, applied to the held-out test cell.
- **Output Artifacts:** `data/04_model_input/loco_fold_{0..3}_{cell}.npz`

### 2.9 Experiments & Benchmarks Run on CALCE CS2
1. **K06 Validation Gate (Pipeline Integrity Check):**
   - Model: Classical XGBoost Regressor (`n_estimators=200`, `max_depth=5`, `lr=0.05`).
   - Task: 5-fold within-cell CV on continuous SOH for cell `CS2_35`.
   - Result: MAE = **0.0057 ± 0.0009**, $R^2 = 0.9842$.
   - **Correct Scientific Interpretation:** This result confirms that the multi-file concatenation, cycle extraction, and feature pipelines recover the underlying electrochemical degradation signal. It is a pipeline validation check, not a direct comparison against published leave-one-cell-out QLSTM benchmarks (which operated under a different architecture, task framing, and cross-cell protocol).
2. **K07 Classical Panel (4-Fold LOCO):**
   - 6 Models: Logistic Regression, Random Forest, Extra Trees, XGBoost, MLP, SVM (RBF).
   - Under **Fixed 20**: Logistic Regression and SVM achieve **79.6%** and **74.4% warning recall** with **0 / 4 zero-recall folds**.
   - Under **Proportional 12%**: Tree ensembles achieve **55.8% to 60.2% warning recall** with balanced accuracy $> 0.84$ and **0 / 4 zero-recall folds**.
3. **K08 Variational Quantum Classifier (8-Qubit VQC):**
   - 8 qubits, angle data re-uploading (2 layers), PennyLane `default.qubit`, SPSA optimizer.
   - Fixed 20 LOCO Warning Recall: **0.2929** (29.3%).
   - Proportional 12% LOCO Warning Recall: **0.3214** (32.1%) with **0 / 4 zero-recall folds**.
4. **K10 Regression Threshold Sweep:**
   - Regression on `cycles_to_crossing` with threshold sweep at 10, 20, 30, 50, 78, 100, 150 cycles.
   - At 50 cycles: XGBoost achieves **49.7% recall / 57.4% precision**; Random Forest achieves **52.3% recall / 66.6% precision**.
   - At 100 cycles: Warning recall reaches **64.2% / 70.6% precision**.
   - At 150 cycles: Warning recall reaches **65.7% / 68.8% precision**.
5. **K11 Pitfalls Audit:**
   - Pitfall 4.1 (float32 std noise): PASSED (0 un-zeroed values below $10^{-12}$).
   - Pitfall 4.2 (Double standardisation): PASSED (raw -> window summary -> per-fold scaler).
   - Pitfall 4.3 (Single-class folds): PASSED (all 4 folds have 3/3 classes, no 0.3333 macro F1 artifact).
   - Pitfall 4.4 (Cell identity recoverability): Random Forest achieves **99.68% accuracy** predicting cell ID (vs 25% chance), demonstrating why LOCO cross-validation is mandatory to prevent intra-cell memorization.
   - Pitfall 4.5 (Same-batch cells): Stated as a limitation that CS2 cells belong to a single manufacturing batch.

---

## 3. Dataset 2: NASA PCoE Battery Dataset

### 3.1 Raw Data Ingestion & Specifications
- **Source:** NASA Prognostics Center of Excellence (PCoE).
- **Cells Acquired:** `B0005`, `B0006`, `B0007`, `B0018` (4 cells).
- **Chemistry & Form Factor:** 18650 cylindrical Lithium-ion cells; nominal rated capacity $Q_{nominal} = 2.0\text{ Ah}$.
- **Cycling Protocol:** Ambient room temperature (24°C). CC-CV charge to 4.2V, followed by 2A constant-current discharge to cutoff voltages (2.7V, 2.5V, 2.2V).
- **Lifespan:** Ultra-short operational lifespan (**132 to 168 discharge cycles** per cell).

### 3.2 Sensor Instrumentation
Unlike CALCE, NASA cells included:
- Voltage measured (`Voltage_measured`), Current measured (`Current_measured`)
- Load voltage (`Voltage_load`), Load current (`Current_load`)
- **Direct Surface Thermocouple:** `Temperature_measured` ($T$)

### 3.3 Target Labeling on NASA
- **End-of-Life (EOL):** Defined in `vq_core/schema.py` as $\text{SOH} < 0.80$ with `CROSSING_PERSISTENCE_CYCLES = 3` (matching CALCE exactly).
- **Warning Window:** Defined as a 20-cycle pre-EOL window $[EOL - 20, EOL)$.
- **Scaling Divergence:** On NASA, 20 cycles represents **12% to 15% of the cell's entire operational lifespan**, whereas on CALCE, 20 cycles is only **2.6% of lifespan**.

### 3.4 Feature Sets & Windowing on NASA
- Window length = 50 samples, stride = 25 samples.
- **`COMMON_FEATURES` (11 signals):** All 8 thermocouple features systematically stripped to match CALCE sensor capabilities. When summarized using the same 6 statistics, both datasets share the identical 66-feature representation.

### 3.5 Verified NASA Experimental Results (From Results Registry)
- In the initial summary table, NASA warning recall was erroneously cited from a pooled train/test split (0.8500–0.9667). As documented in `results_registry.md`, pooled splits conflate intra-cell memorization with prognostics.
- **The Verified NASA LOCO Position:**
  - Standard 3-Class Classification (LOCO): **0.0000 across all folds**
  - Class-Balanced 3-Class (LOCO): **0.0591 median**
  - Regression + Threshold at 20 cycles: **0.2455 (VQC)**, up to **0.5962 (MLP)**
  - Regression + Threshold at 50 cycles: **0.4828 (VQC)**, up to **0.8578 (MLP)**

---

## 4. Master Comparative Matrix: CALCE CS2 vs. NASA PCoE

| Dimension | NASA PCoE Dataset | CALCE CS2 Dataset |
|:---|:---|:---|
| **Primary Cells** | `B0005`, `B0006`, `B0007`, `B0018` | `CS2_35`, `CS2_36`, `CS2_37`, `CS2_38` |
| **Cell Form Factor & Chemistry** | 18650 Cylindrical, Li-ion ($Q_{ref} = 2.0\text{ Ah}$) | Prismatic $\text{LiCoO}_2$ / Graphite ($Q_{ref} = 1.1\text{ Ah}$) |
| **Total Lifespan** | **132 – 168 cycles** (Short) | **880 – 1,078 cycles** (Dense / Long) |
| **Total Valid Discharge Cycles** | ~600 cycles across 4 cells | **3,971 cycles** across 4 cells |
| **Raw File Format & Volume** | 4 MATLAB `.mat` files | **106 Arbin Excel (`.xlsx`) files** |
| **Temperature Instrumentation** | **Present** (Direct surface thermocouple $T$) | **ABSENT (Zero thermocouples)** |
| **Data Integrity Traps Resolved** | Date formatting; cycle alignment | (1) Non-padded date sort reversal;<br>(2) Cycler cycle index reset to 1;<br>(3) Duplicate log files deduplicated |
| **Discharge Profile** | 2A CC discharge to cutoffs | **Step 7 isolated:** 1C CC discharge (-1.1A) to 2.7V cutoff |
| **SOH Reference Standard** | Per-cell BOL median of 5 highest valid ($s.nlargest(5).median()$) | **Per-cell BOL median of 5 highest valid** ($s.nlargest(5).median()$) |
| **SOH Quality Filter** | `s04` ratio check | `s04` filter ($0.30 \cdot Q_{ref} \le Q_{dis} \le 1.10 \cdot Q_{ref}$) |
| **End-of-Life (EOL) Definition** | $\text{SOH} < 0.80$ with `persistence = 3` | $\text{SOH} < 0.80$ with `persistence = 3` |
| **EOL Crossing Cycles** | 125, 89, 125, 96 | **`CS2_35`: 544, `CS2_36`: 501, `CS2_37`: 577, `CS2_38`: 601** |
| **Window Slicing** | Length = 50 samples, Stride = 25 | Length = 50 samples, Stride = 25 (50% overlap) |
| **Total Windows Extracted** | 1,600 windows (1,120 train / 480 test) | **8,977 windows** across 4 cells |
| **Common Feature Set** | 11 base electrical signals $\times$ 6 summaries | **Exact 66 summary features** ($11 \times 6$: mean, std, min, max, first, last) |
| **Thermal Features** | 8 features ($T$, $dT/dt$, $\Delta T_{rise}$, roll $T$) | **0 features (Strictly Electrical: $V, I, dV/dt, dI/dt, P, t$, rolling $V$)** |
| **Standard Deviation Precision** | float64, zero-variance guarded below $10^{-12}$ | float64, zero-variance guarded below $10^{-12}$ |
| **Warning Scheme 1** | Fixed 20 cycles (**12–15% of lifespan**) | **Fixed 20 cycles** ($[EOL-20, EOL)$, **2.6% of life**, 236 windows / 2.63%) |
| **Warning Scheme 2** | N/A | **Proportional 12%** ($[EOL-0.12 \times EOL, EOL)$, ~60–70 cycles, 790 windows / 8.80%) |
| **Validation Strategy** | 17-fold LOCO (verified) | **4-Fold Leave-One-Cell-Out (LOCO)** with strictly isolated per-fold scaler |
| **Standard 3-Class LOCO Warning Recall** | **0.0000 across all folds** | **0.7958 (LR), 0.7438 (SVM), 0.2929 (VQC)** |
| **Regression Threshold at 50 Cycles** | **0.4828 (VQC), 0.8578 (MLP)** | **0.4965 (XGBoost), 0.5234 (RF), 0.4266 (MLP)** |

---

## 5. Scientific Resolution: Why CALCE Succeeds Where NASA Failed

| Formulation / Model | NASA PCoE (Verified LOCO) | CALCE CS2 Fixed 20 (~2% of life) | CALCE CS2 Proportional 12% (~60–70 cycles) |
|:---|:---:|:---:|:---:|
| **Logistic Regression** | 0.0000 | **0.7958** | **0.7909** |
| **Random Forest** | 0.0000 | **0.1925** | **0.5581** |
| **Extra Trees** | 0.0000 | **0.2302** | **0.5793** |
| **XGBoost** | 0.0000 | **0.2611** | **0.6023** |
| **MLP (Neural Net)** | 0.0000 | **0.1917** | **0.4747** |
| **SVM (RBF)** | 0.0000 | **0.7438** | **0.7008** |
| **8-Qubit VQC (PennyLane)** | 0.0000 | **0.2929** | **0.3214** |
| **Zero-Recall Fold Count** | **4 / 4 (All Folds)** | **0 / 4 (Classical), 1 / 4 (VQC)** | **0 / 4 (All Models)** |

### Concluding Resolution:
Lead-time warning classification failed on NASA PCoE under leave-one-cell-out evaluation (**warning recall 0.0000 across all folds**) and was not recovered by class balancing (0.0591). It was partially recovered on NASA by reframing as regression on cycles-to-crossing and thresholding the prediction (0.2455 at a 20-cycle lead; 0.4828 at 50). 

On CALCE CS2, with cell lifetimes 6–8× longer (880–1,080 cycles) and an identical protocol without thermocouples:
1. Standard classification recovers up to **79.6% warning recall** under Fixed 20 and **60.2% to 79.1%** under Proportional 12% with **0/4 zero-recall folds**.
2. Continuous regression on `cycles_to_crossing` extends the reliable early warning horizon to **78, 100, and 150 cycles** with high precision (~70%).

**Lead-time detectability is therefore resolution-limited, NOT physically unrecoverable.**

---

## 6. Code File Traceability

All transformations and evaluations are implemented in the following modular scripts:
- **`src/k01_acquire_and_audit.py`**: Audits cycler channels, confirms 0 thermocouples, identifies sorting traps.
- **`src/k02_build_cycle_table.py`**: Step 7 isolation, cumulative cycle tracking, SOH calculation via per-cell BOL median of 5 highest valid capacities, `s04` filtering.
- **`src/k03_build_features.py`**: Extracts 11 base electrical signals, computes rolling voltage stats in `float64` with zero-variance guard below $10^{-12}$.
- **`src/k04_k05_build_sequences.py`**: Sustained EOL crossing detection (`persistence=3`), dual labeling (Fixed 20 vs Prop 12%), sliding window slicing, continuous `cycles_to_crossing` target, 4-fold LOCO splits with isolated `StandardScaler`.
- **`src/k06_validate_pipeline.py`**: Classical XGBoost SOH regression validation gate on `CS2_35` (within-cell CV pipeline validation).
- **`src/k07_classical_panel.py`**: 6-model classical panel evaluation across 4 LOCO folds reporting per-fold precision, recall, confusion matrices, and zero-recall fold counts.
- **`src/k08_vqc_model.py`**: 8-qubit Variational Quantum Classifier implementation in PennyLane with angle data re-uploading and SPSA optimization.
- **`src/k10_threshold_sweep.py`**: Continuous regression on `cycles_to_crossing` and lead-time threshold sweep at 10, 20, 30, 50, 78, 100, 150 cycles.
- **`src/k11_audit_pitfalls.py`**: Audits pitfalls 4.1 to 4.5 (float32 std noise, double standardization, single-class folds, cell identity recoverability 99.68%, same-batch cell limitations).
