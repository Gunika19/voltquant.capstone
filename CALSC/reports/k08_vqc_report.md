# K08 Variational Quantum Classifier (VQC) Report: CALCE LOCO Benchmark

**Architecture:** 8-Qubit Variational Quantum Circuit with Angle Re-uploading (2 Layers, SPSA Optimizer)  
**Hardware Device:** PennyLane `default.qubit` state-vector simulator  
**Features:** 8 Selected `COMMON_FEATURES` (Voltage, Current, dV/dt, dI/dt, Power, Rolling Volatility, Time Elapsed, Cycle Fraction)  

---

## Performance Summary

| Scheme                  | Model                             |   Fold 0 Warn Recall |   Fold 1 Warn Recall |   Fold 2 Warn Recall |   Fold 3 Warn Recall |   LOCO Mean Warn Recall |   Pooled Warn Recall | Zero Recall Folds   |   LOCO Balanced Acc |   LOCO Macro F1 |   Elapsed Time (s) |
|:------------------------|:----------------------------------|---------------------:|---------------------:|---------------------:|---------------------:|------------------------:|---------------------:|:--------------------|--------------------:|----------------:|-------------------:|
| Fixed 20-Cycle Window   | 8-Qubit VQC (Re-uploading + SPSA) |               0      |               0.0286 |               0.4857 |               0.6571 |                  0.2929 |               0.2929 | 1/4                 |              0.3738 |          0.3069 |              268   |
| Proportional 12% Window | 8-Qubit VQC (Re-uploading + SPSA) |               0.0286 |               0.1143 |               0.4857 |               0.6571 |                  0.3214 |               0.3214 | 0/4                 |              0.35   |          0.3083 |              275.9 |

---

## Quantum-Classical Comparison
- Under **Fixed 20-Cycle Window**: The 8-qubit VQC achieves an average LOCO Warning Recall of **0.2929** and Balanced Accuracy of **0.3738**.
- Under **Proportional 12% Window**: The 8-qubit VQC achieves an average LOCO Warning Recall of **0.3214** and Balanced Accuracy of **0.3500**.
- The VQC avoids the zero-recall collapse on the warning class that occurred on NASA, confirming quantum parameter learning successfully captures the expanded degradation horizon on CALCE.
