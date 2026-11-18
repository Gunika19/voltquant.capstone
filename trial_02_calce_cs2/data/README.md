# CALCE CS2 Dataset (Prismatic LCO / Graphite)

This directory contains the processed data tables, rolling-window features, and model-input sequences for the 4 CALCE CS2 prismatic cells (`CS2_35`, `CS2_36`, `CS2_37`, `CS2_38`).

---

## Data Pipeline Stages

```
trial_02_calce_cs2/data/
├── 01_raw/                    # Downloaded CALCE Excel files (CS2_35 to CS2_38)
├── 02_intermediate/           # Parsed discharge cycle tables (calce_discharge_cycles.parquet)
├── 03_features/               # 66 base temporal & rolling-window degradation features
└── 04_model_input/            # 8 physics-guided early features & 4-fold LOCO splits
```

*Note: Raw binary files (.zip and large parquets, total 374MB) are excluded from Git tracking via `.gitignore`. To re-download and re-extract the raw CALCE Excel files, run `python src/download_remaining_cells.py`.*

