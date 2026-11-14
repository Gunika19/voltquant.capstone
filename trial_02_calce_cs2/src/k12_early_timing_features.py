"""K12 · Enhanced Feature Engineering Pipeline (Electrochemical Domain Observables)

Builds an enriched feature set adding strictly observable electrochemical signals to the
66 baseline window summary features without any label leakage:
1. Ohmic drop proxy: (4.2 - voltage_first) / (|current_first| + eps)
2. Dynamic impedance proxy: (4.2 - voltage_mean) / (|current_mean| + eps)
3. Partial discharge energy: |power_mean| * time_elapsed_last / 3600.0 (Wh)
4. Voltage discharge curvature: voltage_last - 2.0 * voltage_mean + voltage_first
5. Voltage discharge slope: (voltage_last - voltage_first) / (time_elapsed_last - time_elapsed_first + 1.0)
6. Relative window temporal position: time_elapsed_mean / 3600.0
7. Cycle odometer index: cycle_index (observable by standard BMS cycle counter)
8. Voltage rate volatility: dV_dt_std / (|dV_dt_mean| + 1e-6)

Outputs:
- data/04_model_input/calce_window_features_enriched.parquet
- Saves enriched 4-fold LOCO splits in data/04_model_input/loco_enriched_fold_*.npz
"""

from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent

CANDIDATE_DATA_DIRS = [
    BASE_DIR / "data" / "04_model_input",
    Path("trial_02_calce_cs2/data/04_model_input"),
    Path("CALSC/data/04_model_input"),
    Path("data/04_model_input"),
]
DATA_DIR = next((p for p in CANDIDATE_DATA_DIRS if p.exists()), BASE_DIR / "data" / "04_model_input")
DATA_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PARQUET = DATA_DIR / "calce_window_features.parquet"
OUTPUT_PARQUET = DATA_DIR / "calce_window_features_enriched.parquet"
CELLS = ["CS2_35", "CS2_36", "CS2_37", "CS2_38"]

# Non-feature metadata and targets
METADATA_AND_TARGET_COLS = [
    "battery_id",
    "cycle_index",
    "window_start_idx",
    "label_fixed_20",
    "label_prop",
    "soh",
    "cycles_to_crossing",
]

def build_enriched_features() -> pd.DataFrame:
    print(f"Loading baseline window features from {INPUT_PARQUET}...")
    df = pd.read_parquet(INPUT_PARQUET)
    print(f"Loaded {len(df):,} windows with {len(df.columns)} columns.")

    # 1. BMS cycle odometer
    df["cycle_odometer"] = df["cycle_index"].astype(np.float64)

    # 2. Ohmic voltage drop proxy at start of discharge
    df["ohmic_drop_start"] = (4.2 - df["voltage_first"]).astype(np.float64)

    # 3. Dynamic internal resistance proxy (Ohm)
    curr_mag = df["current_mean"].abs() + 1e-6
    df["resistance_proxy_ohm"] = ((4.2 - df["voltage_mean"]) / curr_mag).astype(np.float64)

    # 4. Partial discharge energy proxy (Wh)
    df["partial_energy_Wh"] = (df["power_mean"].abs() * df["time_elapsed_last"] / 3600.0).astype(np.float64)

    # 5. Discharge curve profile curvature
    df["voltage_curvature"] = (df["voltage_last"] - 2.0 * df["voltage_mean"] + df["voltage_first"]).astype(np.float64)

    # 6. Voltage plateau slope (V/s)
    time_span = df["time_elapsed_last"] - df["time_elapsed_first"] + 1.0
    df["voltage_plateau_slope"] = ((df["voltage_last"] - df["voltage_first"]) / time_span).astype(np.float64)

    # 7. Normalized intra-cycle temporal progress (0 = start, 1 = full 1-hour discharge)
    df["intra_cycle_progress"] = (df["time_elapsed_mean"] / 3600.0).clip(0.0, 1.5).astype(np.float64)

    # 8. Voltage rate relative volatility
    df["dV_dt_volatility"] = (df["dV_dt_std"] / (df["dV_dt_mean"].abs() + 1e-6)).clip(-100.0, 100.0).astype(np.float64)

    # Verification: check for NaNs or Infs
    feature_cols = [c for c in df.columns if c not in METADATA_AND_TARGET_COLS]
    print(f"Total engineered feature count: {len(feature_cols)} (66 baseline + 8 electrochemical observables = 74)")
    assert df[feature_cols].isna().sum().sum() == 0, "Enriched features contain NaN values!"
    assert np.isinf(df[feature_cols].values).sum() == 0, "Enriched features contain Inf values!"

    df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"Saved enriched master table to {OUTPUT_PARQUET}")

    # Build and save enriched LOCO folds
    print("\nGenerating 4-Fold Enriched LOCO splits...")
    for fold_idx, test_cell in enumerate(CELLS):
        train_cells = [c for c in CELLS if c != test_cell]
        train_df = df[df["battery_id"].isin(train_cells)].copy()
        test_df = df[df["battery_id"] == test_cell].copy()

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[feature_cols].values)
        X_test = scaler.transform(test_df[feature_cols].values)

        out_npz = DATA_DIR / f"loco_enriched_fold_{fold_idx}_{test_cell}.npz"
        np.savez_compressed(
            out_npz,
            X_train=X_train,
            X_test=X_test,
            y_train_fixed=train_df["label_fixed_20"].values,
            y_test_fixed=test_df["label_fixed_20"].values,
            y_train_prop=train_df["label_prop"].values,
            y_test_prop=test_df["label_prop"].values,
            cycles_to_crossing_train=train_df["cycles_to_crossing"].values,
            cycles_to_crossing_test=test_df["cycles_to_crossing"].values,
            train_cells=np.array(train_cells),
            test_cell=np.array(test_cell),
            feature_names=np.array(feature_cols),
            window_start_idx_test=test_df["window_start_idx"].values,
            cycle_index_test=test_df["cycle_index"].values,
        )
        print(f"  Fold {fold_idx} ({test_cell}): Train N={len(train_df)}, Test N={len(test_df)} -> {out_npz.name}")

    return df

if __name__ == "__main__":
    t0 = time.time()
    build_enriched_features()
    print(f"\nK12 complete in {time.time() - t0:.1f}s.")

