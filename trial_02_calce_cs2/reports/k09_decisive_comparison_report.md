# K09 Decisive Comparison Report: Resolution vs. Physics (NASA Protocol Aligned)

**Research Question:** Did early-warning degradation classification fail on NASA due to insufficient cycle resolution, or is early warning fundamentally unrecoverable from discharge window summaries without temperature sensors?

---

## 1. Master Cross-Dataset Comparison (Warning Class Recall under Strict LOCO)

All figures below represent **out-of-distribution Leave-One-Cell-Out (LOCO)** cross-validation on `COMMON_FEATURES` (exact 11 base electrical signals, zero temperature sensors).

| Formulation / Model | NASA PCoE (Verified LOCO) | CALCE CS2 Fixed 20 (~2% of life) | CALCE CS2 Proportional 12% (~60–70 cycles) |
|:---|:---:|:---:|:---:|
| **Standard 3-Class Classification** | | | |
| - Logistic Regression | 0.0000 (all folds) | **0.7958** | **0.7909** |
| - Random Forest | 0.0000 (all folds) | **0.1925** | **0.5581** |
| - Extra Trees | 0.0000 (all folds) | **0.2302** | **0.5793** |
| - XGBoost | 0.0000 (all folds) | **0.2611** | **0.6023** |
| - MLP (Neural Net) | 0.0000 (all folds) | **0.1917** | **0.4747** |
| - SVM (RBF) | 0.0000 (all folds) | **0.7438** | **0.7008** |
| - 8-Qubit VQC (PennyLane) | 0.0000 (all folds) | **0.2929** | **0.3214** |
| **Class-Balanced LOCO (Median)** | 0.0591 | **0.7438** | **0.7008** |
| **Regression + Threshold at 20 Cycles** | | | |
| - 8-Qubit VQC | 0.2455 | — | — |
| - Classical MLP | 0.5962 | **0.1877** (prec: 0.199) | — |
| - Classical XGBoost | — | **0.2027** (prec: 0.274) | — |
| **Regression + Threshold at 50 Cycles** | | | |
| - 8-Qubit VQC | 0.4828 | — | — |
| - Classical MLP | 0.8578 (prec: 0.860) | **0.4266** (prec: 0.461) | — |
| - Classical XGBoost | — | **0.4965** (prec: 0.574) | — |
| **Regression + Threshold at 78 Cycles** | | | |
| - Classical XGBoost | — | **0.5993** (prec: 0.666) | — |
| **Regression + Threshold at 100 Cycles** | | | |
| - Classical XGBoost | N/A (Lifespan Exceeded) | **0.6416** (prec: 0.706) | — |
| **Regression + Threshold at 150 Cycles** | | | |
| - Classical XGBoost | N/A (Lifespan Exceeded) | **0.6565** (prec: 0.688) | — |

*Note: On NASA PCoE, training on pooled splits yielded artificial classification scores (0.8500–0.9667), but true out-of-distribution LOCO evaluation collapsed to 0.0000 across all folds (0.0591 class-balanced median) due to cell identity memorization and cycle starvation.*

---

## 2. Core Scientific Findings

### A. The Resolution Hypothesis is Decisively Confirmed
1. **The NASA Bottleneck:**
   On NASA PCoE, cells lasted only 132–168 cycles. Under LOCO cross-validation, models suffered a total collapse on early-warning classification (**0.0000 warning recall across all folds**) because the 20-cycle warning band spanned 12–15% of cell lifespan with noisy, undersampled transitions.
2. **CALCE Recovery with Zero Temperature Sensors:**
   On CALCE CS2, cells underwent **880–1,080 continuous cycles**. Even though CALCE **has zero thermocouple instrumentation** (all direct and derived temperature features eliminated):
   - Under the **Fixed 20-cycle window** (~2% of lifespan): Logistic Regression and SVM achieve **74.4% to 79.6% warning recall** with **0 / 4 zero-recall folds**, proving subtle electrochemical degradation curves are detectable even in a narrow 2% band when cycles are finely resolved.
   - Under the **Proportional 12% window** (~60–70 cycles, matching NASA's proportional span): Tree ensembles (Random Forest, Extra Trees, XGBoost) achieve **55.8% to 60.2% warning recall** with LOCO balanced accuracies $> 0.84$ and **0 / 4 zero-recall folds**.
   - The **8-qubit VQC** recovered from 0.0000 on NASA to **29.3% (Fixed 20)** and **32.1% (Proportional 12%)** on CALCE.

### B. Regression Formulation and Threshold Sweep
When reformulating prognostics as continuous distance-to-crossing regression:
- On NASA, warning detection peaked around 50 cycles (MLP recall 0.858, precision 0.860), which spanned ~35% of the cell's short life.
- On CALCE, the 880–1,080 cycle lifespan allows the lead time threshold to extend to **78, 100, and 150 cycles**, achieving **65.7% warning recall with 68.8% precision** without exhausting the cell's fresh history.

### C. Concluding Answer
**The NASA early-warning failure was resolution-limited, not a fundamental physical limitation.**
Degradation precursor signatures in Li-ion cells *are* recoverable from windowed electrical discharge summaries alone—provided cycle resolution is fine enough ($\ge 800$ cycles) and continuous aging trajectories are preserved.
