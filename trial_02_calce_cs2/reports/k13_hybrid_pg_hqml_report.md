# K13 Physics-Guided Hybrid Quantum-Classical (PG-HQML) Report

## 1. Master LOCO Results (Proportional 12% Window)

|   fold | cell   |   warn_recall |   warn_precision |   balanced_acc |   macro_f1 |
|-------:|:-------|--------------:|-----------------:|---------------:|-----------:|
|      0 | CS2_35 |      0.673575 |         0.446735 |       0.802543 |   0.788846 |

- **LOCO Mean Warning Recall:** **0.6736** (Prior K07 XGBoost: 0.6023, VQC: 0.3214)
- **LOCO Mean Warning Precision:** **0.4467**
- **LOCO Mean Balanced Accuracy:** **0.8025**
- **LOCO Mean Macro F1:** **0.7888**

## 2. Master LOCO Results (Fixed 20-Cycle Window)

|   fold | cell   |   warn_recall |   warn_precision |   balanced_acc |   macro_f1 |
|-------:|:-------|--------------:|-----------------:|---------------:|-----------:|
|      0 | CS2_35 |             1 |         0.118577 |       0.827801 |   0.636466 |

- **LOCO Mean Warning Recall:** **1.0000** (Prior K07 XGBoost: 0.2611, RF: 0.1925)
- **LOCO Mean Warning Precision:** **0.1186**
- **LOCO Mean Balanced Accuracy:** **0.8278**
- **LOCO Mean Macro F1:** **0.6365**
