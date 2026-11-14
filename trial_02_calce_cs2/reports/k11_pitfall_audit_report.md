# K11 Pitfall Audit Report: CALCE Pipeline Rigor & NASA Protocol Alignment

**Date:** September 2026  
**Dataset:** CALCE CS2 (`CS2_35`, `CS2_36`, `CS2_37`, `CS2_38`)  

## 1. Pitfall 4.1: float32 Standard Deviation & Rounding Noise
- **Status:** **PASSED**
- Total `_std` feature values audited across 11 columns: 98,747
- Count of non-zero standard deviations below $10^{-12}$: **0**
- All standard deviations are computed in `float64` with strict zero-variance clipping at $< 10^{-12}$.

---

## 2. Pitfall 4.2: Order of Feature Extraction and Scaling
- **Status:** **PASSED**
- Pipeline order verified: `Raw Physical Measurements` $	o$ `Window Summary Statistics` $	o$ `Per-Fold StandardScaler`.
- Global scaling before window summarisation is strictly prohibited and was not performed.

---

## 3. Pitfall 4.3: Class Counts Across 4 LOCO Folds
Checking whether any held-out fold contains fewer than 3 classes (which would trigger trivial $0.3333$ macro F1 artifacts under `zero_division=0`):

| Fold            | Fixed 20 (N/W/D)   |   Fixed Distinct Classes | Prop 12% (N/W/D)   |   Prop Distinct Classes |
|:----------------|:-------------------|-------------------------:|:-------------------|------------------------:|
| Fold 0 (CS2_35) | 1574 / 60 / 508    |                        3 | 1441 / 193 / 508   |                       3 |
| Fold 1 (CS2_36) | 1449 / 60 / 578    |                        3 | 1329 / 180 / 578   |                       3 |
| Fold 2 (CS2_37) | 1669 / 58 / 591    |                        3 | 1524 / 203 / 591   |                       3 |
| Fold 3 (CS2_38) | 1740 / 58 / 632    |                        3 | 1584 / 214 / 632   |                       3 |

- **Audit Result:** **PASSED (All 4 folds have 3/3 classes)**
- Unlike NASA where 6 of 17 folds were single-class (collapsing macro F1 to 0.3333), all 4 CALCE cells complete a full lifetime from fresh to EOL crossing and deep degradation, ensuring every fold contains Normal, Warning, and Degraded classes.

---

## 4. Pitfall 4.4: Cell Identity Recoverability Audit
- **Chance Accuracy (4 cells):** **25.0%**
- **Random Forest Cell ID Classification Accuracy:** **99.68%**
- **Mechanism Analysis:**
  On NASA, Random Forest recovered cell ID with 86% accuracy (chance 5.9%) even without protocol covariates, explaining the large gap between pooled and LOCO evaluation.
  On CALCE, cell identity recoverability is **99.7%**. Because CALCE cells were tested under identical temperature (ambient) and identical C-rate (1C CC discharge), the recoverability reflects subtle cell-to-cell manufacturing variations, confirming that LOCO cross-validation is essential to prevent intra-cell memorization.

---

## 5. Pitfall 4.5: Same-Batch Grouping & Stated Limitations
- **Audit Finding:** Cells `CS2_35`, `CS2_36`, `CS2_37`, and `CS2_38` belong to the same manufacturing batch of prismatic 1.1 Ah $\text{LiCoO}_2$ cells tested under continuous 1C cycling at the University of Maryland CALCE center.
- **Research Limitation:** Because these cells share a common batch and identical cycling protocol, the 4 LOCO folds represent variations in operational degradation rather than distinct operating temperatures or load profiles. This limitation is explicitly stated in the paper's threat-to-validity section.
