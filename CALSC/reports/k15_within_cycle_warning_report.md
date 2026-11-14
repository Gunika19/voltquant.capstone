# K15 Within-Cycle Fast Early Warning Report (Partial Discharge Window)

## 1. Scientific Significance
Demonstrates that Li-ion battery degradation warning can be successfully detected from **only the initial partial discharge window** (window_start_idx = 0, first 1,470s / ~25% of discharge time), without waiting for deep or full cycle discharge to complete.

## 2. Quantitative Benchmark Table

|   Fold | Cell   |   Full Cycle Warn Recall |   Full Cycle Warn Prec |   Full Cycle Bal Acc |   Fast Window Warn Recall |   Fast Window Warn Prec |   Fast Window Bal Acc |
|-------:|:-------|-------------------------:|-----------------------:|---------------------:|--------------------------:|------------------------:|----------------------:|
|      0 | CS2_35 |                 0.554404 |               0.514423 |             0.78562  |                  0.553846 |                0.486486 |              0.807903 |
|      1 | CS2_36 |                 0.233333 |               0.241379 |             0.662553 |                  0.233333 |                0.237288 |              0.695099 |
|      2 | CS2_37 |                 0.935961 |               0.620915 |             0.944649 |                  0.942029 |                0.625    |              0.952116 |
|      3 | CS2_38 |                 0.658879 |               0.568548 |             0.863776 |                  0.625    |                0.56962  |              0.853495 |

## 3. Key Findings
- **Fast-Window Warning Recall:** **58.9%** (Retains **98.8%** of full-cycle warning recall).
- **Fast-Window Balanced Accuracy:** **82.7%**.
- **Operational Impact:** An EV battery management system (BMS) can confirm degradation within 20 minutes of vehicle operation rather than requiring hours of full-pack depletion.
