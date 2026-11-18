# MIT Battery Cycle-Life Dataset: Data Staging Directory

This directory is the staging area for the **MIT / Stanford / Toyota Research Institute** fast-charging commercial battery cohort (Severson et al., *Nature Energy* 4, 383–391, 2019).

---

## Required Files to Drop Here

When uploading your MIT battery files, place them directly into this folder (`MIT/data/`):

| File Name | Description | Required / Optional |
|---|---|---|
| `mit_cycle_life_clean.csv` | **Primary Model Input.** 124 cells $\times$ 20 Severson features computed over cycles 2–100 after applying data-quality rules R1, R2, R3. | **Required** |
| `mit_cycle_life_raw.csv` | Raw 20-feature matrix without data-cleaning rules (for sensitivity & robustness checks). | Optional |
| `mit_splits_70_30_x20.csv` | 20 random 70/30 train/test splits (seeds 0–19) stratified into 5 log-lifetime quantiles (protocol of Fei et al. 2021). | Optional / Reference |
| `FEATURES.md` | Feature definitions, mathematical formulas, and cycle ranges for all 20 early features. | Recommended |
| `mit_data_quality_log.csv` | Audit trail of cycles affected by R1 (impossible capacity), R2 (paused cycles), and R3 (stuck sensor). | Optional |

---

## Dataset Schema & Key Columns

Each row in `mit_cycle_life_clean.csv` represents one commercial A123 APR18650M1A LFP cell:

* **Identifiers & Target:**
  * `cell_id`: Unique cell identifier (e.g., `b1c0` to `b3c45`).
  * `batch`: Experimental batch (`b1` = 2017-05-12, `b2` = 2017-06-30, `b3` = 2018-04-12).
  * `charge_policy`: Fast-charging protocol (e.g., `3.6C(80%)-3.6C`, `4C(80%)-4C`, etc.).
  * `severson_split`: Severson's official published split (`train` = 41 cells, `primary_test` = 43 cells, `secondary_test` = 40 cells).
  * `cycle_life`: True cycle life (cycles to 80% nominal capacity = 0.88 Ah, range: 148 to 2,237 cycles).
  * `log10_cycle_life`: $\log_{10}(\text{cycle life})$ target used for regression training.

* **20 Early Prognostic Features (Cycles 2 to 100):**
  1. `dq_min`: Minimum of $\Delta Q_{100-2}(V)$ curve.
  2. `dq_mean`: Mean of $\Delta Q_{100-2}(V)$ curve.
  3. `dq_var`: **Log variance of $\Delta Q_{100-2}(V)$** (single strongest predictor, $r = -0.927$).
  4. `dq_skew`: Skewness of $\Delta Q_{100-2}(V)$ curve.
  5. `dq_kurt`: Kurtosis of $\Delta Q_{100-2}(V)$ curve.
  6. `dq_2v`: $\Delta Q(V)$ evaluated at $V = 2.0\text{ V}$.
  7. `cap_slope_2_100`: Slope of discharge capacity fade from cycle 2 to 100.
  8. `cap_intercept_2_100`: Intercept of linear fit for capacity fade (cycles 2–100).
  9. `cap_slope_91_100`: Slope of discharge capacity fade over the last 10 cycles (cycles 91–100).
  10. `cap_intercept_91_100`: Intercept of linear fit for capacity fade (cycles 91–100).
  11. `q_cycle2`: Discharge capacity at cycle 2 (Ah).
  12. `qmax_minus_q2`: Max capacity observed in cycles 2–100 minus cycle 2 capacity.
  13. `q_cycle100`: Discharge capacity at cycle 100 (Ah).
  14. `charge_time_avg`: Average charging duration over cycles 2–100 (minutes).
  15. `temp_max`: Maximum recorded temperature in cycles 2–100 ($^\circ\text{C}$).
  16. `temp_min`: Minimum recorded temperature in cycles 2–100 ($^\circ\text{C}$).
  17. `temp_integral`: Integrated temperature-time product over cycles 2–100.
  18. `ir_cycle2`: Internal resistance at cycle 2 ($\Omega$).
  19. `ir_min`: Minimum internal resistance recorded in cycles 2–100 ($\Omega$).
  20. `ir_diff_100_2`: Resistance change between cycle 100 and cycle 2 ($\Omega$).

---

## Known Missing Values & Imputation Handling

- **Missing values in `mit_cycle_life_clean.csv`:**
  - Exactly **2 blank values**: Cell `b1c18` $\to$ `ir_cycle2` and `ir_diff_100_2`.
  - Reason: Cell `b1c18` experienced a frozen internal resistance sensor starting at cycle 2. Under Rule R3, repeated readings were purged rather than simulated.
- **Handling in VoltQuant pipeline:**
  - Automated imputation replaces `b1c18` missing values with the **Batch 1 median** for those specific resistance metrics during training, ensuring zero dropped cells and zero runtime errors in PyTorch/XGBoost.

