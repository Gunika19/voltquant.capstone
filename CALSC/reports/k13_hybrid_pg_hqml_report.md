# K13 Physics-Guided Hybrid Quantum-Classical (PG-HQML) Report

## 1. Master LOCO Results (Proportional 12% Window)

|   fold | cell   |   warn_recall |   warn_precision |   balanced_acc |   macro_f1 |
|-------:|:-------|--------------:|-----------------:|---------------:|-----------:|
|      0 | CS2_35 |      0.673575 |         0.446735 |       0.802967 |   0.78915  |
|      1 | CS2_36 |      0.566667 |         0.432203 |       0.77572  |   0.774723 |
|      2 | CS2_37 |      0.950739 |         0.543662 |       0.936061 |   0.869795 |
|      3 | CS2_38 |      0.934579 |         0.621118 |       0.950301 |   0.897359 |

- **LOCO Mean Warning Recall:** **0.7814** (Prior K07 XGBoost: 0.6023, VQC: 0.3214)
- **LOCO Mean Warning Precision:** **0.5109**
- **LOCO Mean Balanced Accuracy:** **0.8663**
- **LOCO Mean Macro F1:** **0.8328**

## 2. Master LOCO Results (Fixed 20-Cycle Window)

|   fold | cell   |   warn_recall |   warn_precision |   balanced_acc |   macro_f1 |
|-------:|:-------|--------------:|-----------------:|---------------:|-----------:|
|      0 | CS2_35 |      1        |         0.121212 |       0.826554 |   0.636504 |
|      1 | CS2_36 |      1        |         0.136054 |       0.853551 |   0.661951 |
|      2 | CS2_37 |      0.931034 |         0.122727 |       0.877697 |   0.683741 |
|      3 | CS2_38 |      0.965517 |         0.12095  |       0.876615 |   0.676717 |

- **LOCO Mean Warning Recall:** **0.9741** (Prior K07 XGBoost: 0.2611, RF: 0.1925)
- **LOCO Mean Warning Precision:** **0.1252**
- **LOCO Mean Balanced Accuracy:** **0.8586**
- **LOCO Mean Macro F1:** **0.6647**
