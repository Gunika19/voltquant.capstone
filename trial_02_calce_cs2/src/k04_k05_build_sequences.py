"""K04 & K05 · Three-Class Label Construction, Sequencing, and LOCO Split

Implements:
- K04: Dual Three-Class Early Warning Labels:
  1. Fixed 20-cycle window (direct NASA comparison, ~2% of CALCE life)
  2. Proportional window (~12% of pre-crossing history, ~60-70 cycles, matching NASA's ratio)
  Persistence = 3 cycles below 0.80 SOH threshold.
- K05: Sequences and Grouped Leave-One-Cell-Out (LOCO) Split:
  Length 50, stride 25, within discharge cycles only (never spanning boundaries).
  4-Fold LOCO: each cell held out as test in one fold.
  Scalers fit strictly on train split per fold.
"""

from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from k03_build_features import build_features_for_cell, COMMON_FEATURE_COLUMNS, assert_no_label_leakage

DATA_DIR = Path("data/04_model_input")
DATA_DIR.mkdir(parents=True, exist_ok=True)
CYCLE_TABLE_PATH = Path("data/02_intermediate/calce_discharge_cycles.parquet")

WINDOW_SIZE = 50
STRIDE = 25
CELLS = ["CS2_35", "CS2_36", "CS2_37", "CS2_38"]
SOH_EOL_THRESHOLD = 0.80
CROSSING_PERSISTENCE = 3
WARNING_LEAD_FIXED = 20
PROPORTIONAL_RATIO = 0.12

def compute_sustained_crossings(cycle_table: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Determine sustained crossing cycle and window bounds for each cell."""
    cell_info = {}
    for cell_id, group in cycle_table.groupby("battery_id"):
        valid = group[group["valid"]].sort_values("cycle_index").reset_index(drop=True)
        below = (valid["soh"] < SOH_EOL_THRESHOLD).values
        cross_idx = None
        for i in range(len(below) - CROSSING_PERSISTENCE + 1):
            if np.all(below[i : i + CROSSING_PERSISTENCE]):
                cross_idx = i
                break
                
        if cross_idx is None:
            raise RuntimeError(f"Cell {cell_id} did not reach sustained 0.80 EOL crossing!")
            
        cross_cyc = int(valid.loc[cross_idx, "cycle_index"])
        prop_lead = max(5, int(round(PROPORTIONAL_RATIO * cross_cyc)))
        
        cell_info[cell_id] = {
            "crossing_cycle": cross_cyc,
            "fixed_warning_start": cross_cyc - WARNING_LEAD_FIXED,
            "prop_warning_start": cross_cyc - prop_lead,
            "prop_lead": prop_lead,
            "total_valid_cycles": len(valid)
        }
    return cell_info

def assign_labels(cycle_index: int, info: dict[str, int]) -> tuple[int, int]:
    """Assign (label_fixed, label_prop) for a given cycle."""
    cross_cyc = info["crossing_cycle"]
    
    # 1. Fixed 20
    if cycle_index >= cross_cyc:
        lbl_fixed = 2 # runaway / failed
    elif cycle_index >= info["fixed_warning_start"]:
        lbl_fixed = 1 # warning
    else:
        lbl_fixed = 0 # normal
        
    # 2. Proportional
    if cycle_index >= cross_cyc:
        lbl_prop = 2
    elif cycle_index >= info["prop_warning_start"]:
        lbl_prop = 1
    else:
        lbl_prop = 0
        
    return lbl_fixed, lbl_prop

def extract_all_windows() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract length-50 windows and summary tabular features across all 4 cells."""
    cycle_table = pd.read_parquet(CYCLE_TABLE_PATH)
    cell_crossings = compute_sustained_crossings(cycle_table)
    
    print("\n=== K04 THREE-CLASS LABEL BOUNDARIES ===")
    for c, info in cell_crossings.items():
        print(f"{c}: Sustained EOL crossing at cycle {info['crossing_cycle']}")
        print(f"  Fixed 20 window: cycles [{info['fixed_warning_start']}, {info['crossing_cycle'] - 1}]")
        print(f"  Proportional ({PROPORTIONAL_RATIO*100:.0f}%) window: cycles [{info['prop_warning_start']}, {info['crossing_cycle'] - 1}] (lead = {info['prop_lead']} cycles)")

    tabular_records = []
    manifest_records = []
    
    for cell_id in CELLS:
        print(f"\nProcessing measurements for {cell_id}...")
        df_cell = build_features_for_cell(cell_id)
        info = cell_crossings[cell_id]
        
        window_count_cell = 0
        for cycle_idx, group in df_cell.groupby("cycle_index"):
            group = group.sort_values("time_elapsed").reset_index(drop=True)
            n_samples = len(group)
            if n_samples < WINDOW_SIZE:
                continue
                
            lbl_fixed, lbl_prop = assign_labels(cycle_idx, info)
            soh_val = float(group["soh"].iloc[0])
            
            for start_idx in range(0, n_samples - WINDOW_SIZE + 1, STRIDE):
                end_idx = start_idx + WINDOW_SIZE
                win = group.iloc[start_idx:end_idx]
                
                # Tabular summary representation for classical panel (mean, std, min, max, last)
                row_features = {
                    "battery_id": cell_id,
                    "cycle_index": cycle_idx,
                    "window_start_idx": start_idx,
                    "label_fixed_20": lbl_fixed,
                    "label_prop": lbl_prop,
                    "soh": soh_val,
                }
                
                for col in COMMON_FEATURE_COLUMNS:
                    vals = win[col].values.astype(np.float64)
                    row_features[f"{col}_mean"] = float(np.mean(vals))
                    std_val = float(np.std(vals))
                    row_features[f"{col}_std"] = 0.0 if std_val < 1e-12 else std_val
                    row_features[f"{col}_min"] = float(np.min(vals))
                    row_features[f"{col}_max"] = float(np.max(vals))
                    row_features[f"{col}_first"] = float(vals[0])
                    row_features[f"{col}_last"] = float(vals[-1])
                    
                # Target for Step 5: continuous distance to EOL crossing
                row_features["cycles_to_crossing"] = int(info["crossing_cycle"] - cycle_idx)
                
                tabular_records.append(row_features)
                window_count_cell += 1
                
        print(f"  {cell_id}: generated {window_count_cell:,} windows.")
        
    df_tabular = pd.DataFrame(tabular_records)
    print(f"\nTotal Dataset Extracted: {len(df_tabular):,} windows across all 4 cells.")
    
    # Check class distributions
    print("\nClass Distribution - Fixed 20 Window:")
    print(df_tabular["label_fixed_20"].value_counts(normalize=False))
    print(df_tabular["label_fixed_20"].value_counts(normalize=True).round(4) * 100)
    
    print("\nClass Distribution - Proportional Window:")
    print(df_tabular["label_prop"].value_counts(normalize=False))
    print(df_tabular["label_prop"].value_counts(normalize=True).round(4) * 100)
    
    # Save master tabular window dataset
    tabular_path = DATA_DIR / "calce_window_features.parquet"
    df_tabular.to_parquet(tabular_path, index=False)
    print(f"Saved master tabular dataset to {tabular_path}")
    
    return df_tabular

def prepare_loco_folds(df: pd.DataFrame | None = None):
    """Build 4-Fold LOCO (Leave-One-Cell-Out) scaled splits."""
    tabular_path = DATA_DIR / "calce_window_features.parquet"
    if df is None:
        if tabular_path.exists():
            print(f"Loading existing tabular window dataset from {tabular_path}...")
            df = pd.read_parquet(tabular_path)
        else:
            df = extract_all_windows()
        
    metadata_and_target_cols = [
        "battery_id", "cycle_index", "window_start_idx", 
        "label_fixed_20", "label_prop", "soh", "cycles_to_crossing"
    ]
    feature_cols = [c for c in df.columns if c not in metadata_and_target_cols]
    assert_no_label_leakage(feature_cols)
    print(f"\nStrict COMMON_FEATURES model input count: {len(feature_cols)} (Expected: 66 = 11 signals x 6 stats)")
    assert len(feature_cols) == 66, f"Expected exactly 66 features, got {len(feature_cols)}"
    
    # Build 4 LOCO folds
    for fold_idx, test_cell in enumerate(CELLS):
        train_cells = [c for c in CELLS if c != test_cell]
        
        train_df = df[df["battery_id"].isin(train_cells)].copy()
        test_df = df[df["battery_id"] == test_cell].copy()
        
        # Fit scaler STRICTLY on train split
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[feature_cols].values)
        X_test = scaler.transform(test_df[feature_cols].values)
        
        y_train_fixed = train_df["label_fixed_20"].values
        y_test_fixed = test_df["label_fixed_20"].values
        
        y_train_prop = train_df["label_prop"].values
        y_test_prop = test_df["label_prop"].values
        
        cycles_to_crossing_train = train_df["cycles_to_crossing"].values
        cycles_to_crossing_test = test_df["cycles_to_crossing"].values
        
        out_npz = DATA_DIR / f"loco_fold_{fold_idx}_{test_cell}.npz"
        np.savez_compressed(
            out_npz,
            X_train=X_train,
            X_test=X_test,
            y_train_fixed=y_train_fixed,
            y_test_fixed=y_test_fixed,
            y_train_prop=y_train_prop,
            y_test_prop=y_test_prop,
            cycles_to_crossing_train=cycles_to_crossing_train,
            cycles_to_crossing_test=cycles_to_crossing_test,
            train_cells=np.array(train_cells),
            test_cell=np.array(test_cell),
            feature_names=np.array(feature_cols)
        )
        print(f"Fold {fold_idx}: Test={test_cell} (N={len(test_df)}), Train={train_cells} (N={len(train_df)}) -> saved {out_npz.name}")

if __name__ == "__main__":
    t0 = time.time()
    prepare_loco_folds()
    print(f"\nK04/K05 complete in {time.time() - t0:.1f}s.")
