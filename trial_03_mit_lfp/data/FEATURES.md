# MIT cycle-life dataset — feature definitions (Part 2)

**Data:** values computed only from the original MIT/Stanford/Toyota release
(three `*_batchdata_updated_struct_errorcorrect.mat` files), 124-cell cohort
selected in Part 1 following Severson et al.'s public loading code.

**Formulas:** the 20 candidate features of Severson, K. A., Attia, P. M., et al.
(2019). *Data-driven prediction of battery cycle life before capacity
degradation.* Nature Energy, 4, 383–391. Table 1 / Supplementary Table 1 and
Supplementary Note 1. No other features were added.

**Conventions**
- "Cycle n" uses Severson's 1-based numbering (Python index n−1).
- `log` = log base 10 of the absolute value (Severson, Supplementary Note 1).
- ΔQ₁₀₀₋₁₀(V) = Qdlin(cycle 100) − Qdlin(cycle 10), where Qdlin is discharge
  capacity interpolated on a 1000-point voltage grid from 3.6 V to 2.0 V.
- Skewness / kurtosis follow MATLAB defaults (biased; kurtosis not minus 3).
- Q(n) = summary discharge capacity at cycle n; IR(n), Tmax(n), Tmin(n) and
  chargetime(n) likewise from the per-cycle summary.

| # | Column | Severson feature | Formula used | Severson models |
|---|---|---|---|---|
| 1 | `dq_min` | Minimum of ΔQ₁₀₀₋₁₀(V) | log₁₀\|min ΔQ\| | discharge, full |
| 2 | `dq_mean` | Mean of ΔQ₁₀₀₋₁₀(V) | log₁₀\|mean ΔQ\| | – |
| 3 | `dq_var` | Variance of ΔQ₁₀₀₋₁₀(V) | log₁₀ var(ΔQ) | variance, discharge, full |
| 4 | `dq_skew` | Skewness of ΔQ₁₀₀₋₁₀(V) | log₁₀\|skew ΔQ\| | discharge |
| 5 | `dq_kurt` | Kurtosis of ΔQ₁₀₀₋₁₀(V) | log₁₀\|kurt ΔQ\| | discharge |
| 6 | `dq_2v` | Value of ΔQ₁₀₀₋₁₀(V) at 2 V | log₁₀\|ΔQ(2.0 V)\| | – |
| 7 | `cap_slope_2_100` | Slope of linear fit to capacity fade curve, cycles 2–100 | slope of Q(n) vs n, n = 2…100 | full |
| 8 | `cap_intercept_2_100` | Intercept of that fit, cycles 2–100 | intercept, same fit | full |
| 9 | `cap_slope_91_100` | Slope of linear fit, cycles 91–100 | slope of Q(n) vs n, n = 91…100 | – |
| 10 | `cap_intercept_91_100` | Intercept of linear fit, cycles 91–100 | intercept, same fit | – |
| 11 | `q_cycle2` | Discharge capacity, cycle 2 | Q(2) | discharge, full |
| 12 | `qmax_minus_q2` | Max discharge capacity minus cycle 2 | max Q(n), n = 2…100, minus Q(2) | discharge |
| 13 | `q_cycle100` | Discharge capacity, cycle 100 | Q(100) | – |
| 14 | `charge_time_avg` | Average charge time, first 5 cycles | (1/5) Σ chargetime(n), n = 2…6 | full |
| 15 | `temp_max` | Maximum temperature, cycles 2–100 | max Tmax(n), n = 2…100 | – |
| 16 | `temp_min` | Minimum temperature, cycles 2–100 | min Tmin(n), n = 2…100 | – |
| 17 | `temp_integral` | Integral of temperature over time, cycles 2–100 | Σₙ ∫ T dt (trapezoid on each cycle's T vs t), n = 2…100 | full |
| 18 | `ir_cycle2` | Internal resistance, cycle 2 | IR(2) | – |
| 19 | `ir_min` | Minimum internal resistance, cycles 2–100 | min IR(n), n = 2…100, ignoring IR = 0 (failed reading) | full |
| 20 | `ir_diff_100_2` | Internal resistance, difference between cycle 100 and cycle 2 | IR(100) − IR(2) | full |

**Target:** `cycle_life` (cycles to 80 % of 1.1 Ah nominal capacity) and
`log10_cycle_life` (Severson's models predict the log).

**Splits:** `severson_split` = train (41) / primary_test (43) /
secondary_test (40), as in Severson et al.

**Implementation choices not spelled out in the paper** (so a reader can check them):
- Temperature integral: per-cycle trapezoid of T vs t, summed over cycles 2–100,
  as in MathWorks' published reproduction of Severson's features.
- Minimum IR ignores IR = 0 values (failed readings), also as in that reproduction.
- No cycles are removed for time gaps (MathWorks' example does this; Severson's
  paper does not describe it).

## Two versions: RAW and CLEAN

| File | What it is | Use |
|---|---|---|
| `data/03_primary/mit_features_severson_raw.csv` | The 20 formulas applied to the data exactly as recorded | Faithfulness check / comparison |
| `data/03_primary/mit_features_severson_clean.csv` | The same 20 formulas after rules R1–R3 below | **Main dataset for modelling** |

Both have 124 rows (one per cell) and the same columns.

## Data-quality rules (CLEAN version only)

A read-only scan of all 124 cells, cycles 2–100, found a small number of
recording faults (full table: `data/08_reporting/cell_quality_scan.csv`).
Three rules remove them. Each rule is based on a physical limit or a recorded
fault, and is applied identically to every cell; thresholds are in
`conf/base/parameters_cycle_life_features.yml`.

| Rule | Condition | Action | Why |
|---|---|---|---|
| **R1** Impossible capacity | Discharge capacity > 1.2 Ah | That cycle's capacity reading treated as missing | Cells are rated 1.1 Ah; a reading of 1.5–2.9 Ah cannot be real |
| **R2** Paused cycle | Clock gap > 30 min inside a cycle | Gap skipped when integrating temperature over time (cycle kept) | Testing was paused; idle time is not cycling time |
| **R3** Stuck IR sensor | Same IR value for ≥ 5 consecutive cycles | Those IR readings treated as missing | A real resistance does not repeat to 5 significant digits for weeks |

**Cells affected in the real data** (from the scan; every change is
listed in `data/08_reporting/mit_data_quality_log.csv` after each run):

- **R1:** b1c0 (cycle 12, 1.539 Ah), b1c18 (cycle 40, 2.884 Ah)
- **R2:** one paused cycle (~407 min) in 15 batch-1 cells (b1c0–b1c7, b1c11,
  b1c14–b1c16, b1c20, b1c21; almost identical gap length, so most likely one
  test-wide pause), one in b3c17 (~1,276 min), and 52 cycles in b1c18
- **R3:** b1c18 (IR frozen for 52 consecutive cycles)

**What changes:** only `qmax_minus_q2`, the capacity-fade fits,
`temp_integral`, and b1c18's IR features. The ΔQ₁₀₀₋₁₀(V) features,
`q_cycle2` and `q_cycle100` are not touched by any rule.

**Missing values in CLEAN:** b1c18's `ir_cycle2` and `ir_diff_100_2` are
blank (its cycle-2 IR reading was stuck). They are deliberately **not filled
in** here; how to handle them (imputation, or excluding b1c18 from IR-based
models) is a modelling decision.

**Not treated as faults:** single IR = 0 readings in 10 batch-1 cells
(already ignored by `ir_min`; none falls on cycle 2 or 100), the fast fade of
the shortest-lived cells (b2c1, b2c0, b2c3), and the longer charge times of
batch 1 (different charging policies).

**Relation to Fei et al. (2021) and Anggita & Akrom (2025):** Fei's 42 features
build on these, but their exact definitions were not available to us (paywalled;
authors contacted). We therefore use only Severson's published definitions;
results from those two papers are reported as reference values, not as a
like-for-like comparison.
