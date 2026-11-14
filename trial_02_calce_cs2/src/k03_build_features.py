"""K03 · Feature Engineering (COMMON_FEATURES Only)

Extracts measurement-level features from CALCE Arbin cycler logs for discharge cycles only.
Enforces the COMMON_FEATURES specification:
- Voltage, Current, dV/dt, dI/dt, Power (V * I), time_elapsed
- Rolling voltage statistics (sample-based and time-based 60s, float64 std convention)
- Sampling interval
- Strictly excludes all thermocouple / temperature features (CALCE has no thermocouple)
- Drops label-leakage columns from model input features
"""

from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import pandas as pd

RAW_DIR = Path("data/01_raw/calce")
INTERMEDIATE_DIR = Path("data/02_intermediate")
FEATURES_DIR = Path("data/03_features")
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

CYCLE_TABLE_PATH = INTERMEDIATE_DIR / "calce_discharge_cycles.parquet"

ROLLING_SAMPLES = 5
ROLLING_SECONDS = 60.0

COMMON_FEATURE_COLUMNS = [
    "time_elapsed",
    "voltage",
    "current",
    "dV_dt",
    "dI_dt",
    "power",
    "roll_mean_voltage_samples",
    "roll_std_voltage_samples",
    "roll_mean_voltage_60s",
    "roll_std_voltage_60s",
    "sampling_interval_s",
]

# Guard against label leakage
LABEL_ONLY_COLUMNS = [
    "capacity",
    "capacity_ref",
    "soh",
    "soh_pct",
    "fade_pct",
    "label",
    "cycles_to_crossing",
]

def assert_no_label_leakage(columns: list[str]) -> None:
    """Ensure no SOH/capacity/label column appears in the feature matrix."""
    for col in columns:
        for leak in LABEL_ONLY_COLUMNS:
            if leak.lower() in col.lower():
                raise ValueError(f"Label leakage detected! Column '{col}' matches forbidden pattern '{leak}'.")

def extract_cycle_measurements(df_discharge: pd.DataFrame, cell_id: str, cycle_idx: int, soh: float) -> pd.DataFrame | None:
    """Extract and compute COMMON_FEATURES for a single discharge cycle."""
    if len(df_discharge) < 10:
        return None
        
    df = df_discharge.copy()
    
    # Sort by test time within cycle
    df = df.sort_values("Test_Time(s)").reset_index(drop=True)
    
    # Elapsed time from start of discharge
    t0 = df["Test_Time(s)"].iloc[0]
    df["time_elapsed"] = (df["Test_Time(s)"] - t0).astype(np.float64)
    
    # Voltage, Current, Power
    v = df["Voltage(V)"].astype(np.float64)
    i = df["Current(A)"].astype(np.float64)
    df["voltage"] = v
    df["current"] = i
    df["power"] = (v * i).astype(np.float64)
    
    # Time delta
    dt = df["time_elapsed"].diff().fillna(1.0).astype(np.float64)
    dt = np.where(dt <= 0.0, 1.0, dt)
    df["sampling_interval_s"] = dt
    
    # Finite derivatives dV/dt and dI/dt
    df["dV_dt"] = (v.diff().fillna(0.0) / dt).astype(np.float64)
    df["dI_dt"] = (i.diff().fillna(0.0) / dt).astype(np.float64)
    
    # Sample-based rolling stats (float64 with zero-variance guard)
    df["roll_mean_voltage_samples"] = v.rolling(window=ROLLING_SAMPLES, min_periods=1).mean().astype(np.float64)
    std_samp = v.rolling(window=ROLLING_SAMPLES, min_periods=1).std(ddof=1).fillna(0.0).astype(np.float64)
    df["roll_std_voltage_samples"] = np.where(std_samp < 1e-12, 0.0, std_samp)
    
    # Time-based rolling stats (using time offset index with zero-variance guard)
    temp_time_df = pd.DataFrame({"voltage": v})
    temp_time_df.index = pd.to_timedelta(df["time_elapsed"], unit="s")
    roll_60 = temp_time_df["voltage"].rolling("60s", min_periods=1)
    df["roll_mean_voltage_60s"] = roll_60.mean().values.astype(np.float64)
    std_60 = roll_60.std(ddof=1).fillna(0.0).values.astype(np.float64)
    df["roll_std_voltage_60s"] = np.where(std_60 < 1e-12, 0.0, std_60)
    
    # Attach tracking / label columns (for supervised routing)
    df["battery_id"] = cell_id
    df["cycle_index"] = cycle_idx
    df["soh"] = soh
    
    return df

def build_features_for_cell(cell_id: str, max_cycles: int | None = None) -> pd.DataFrame:
    """Build measurement features for a single cell across all its valid discharge cycles."""
    cycle_table = pd.read_parquet(CYCLE_TABLE_PATH)
    cell_cycles = cycle_table[(cycle_table["battery_id"] == cell_id) & (cycle_table["valid"])].sort_values("cycle_index")
    
    if max_cycles:
        cell_cycles = cell_cycles.head(max_cycles)
        
    print(f"Building features for {cell_id} across {len(cell_cycles)} cycles...")
    
    # Group cycles by source file to minimize file reads
    cycle_measurements = []
    
    for file_name, group in cell_cycles.groupby("source_file"):
        # Locate file
        file_path = next(RAW_DIR.rglob(file_name))
        xl = pd.ExcelFile(file_path)
        sheet = [s for s in xl.sheet_names if "channel" in s.lower()][0]
        df_raw = pd.read_excel(xl, sheet_name=sheet)
        
        for _, row in group.iterrows():
            loc_cyc = row["local_cycle_index"]
            glob_cyc = row["cycle_index"]
            soh = row["soh"]
            
            dis_sub = df_raw[(df_raw["Cycle_Index"] == loc_cyc) & (df_raw["Step_Index"] == 7) & (df_raw["Current(A)"] < -0.2)]
            
            proc_df = extract_cycle_measurements(dis_sub, cell_id, glob_cyc, soh)
            if proc_df is not None:
                cycle_measurements.append(proc_df)
                
    cell_df = pd.concat(cycle_measurements, ignore_index=True)
    print(f"  {cell_id}: Extracted {len(cell_df):,} measurement rows across {len(cycle_measurements)} cycles.")
    return cell_df

if __name__ == "__main__":
    # Test feature extraction on first 20 cycles of CS2_35
    df_test = build_features_for_cell("CS2_35", max_cycles=20)
    print("\nFeature Columns verification:")
    for col in COMMON_FEATURE_COLUMNS:
        print(f"  {col}: mean={df_test[col].mean():.4f}, std={df_test[col].std():.4f}")
    assert_no_label_leakage(COMMON_FEATURE_COLUMNS)
    print("\nLeakage check: PASSED. Zero label-derived columns in COMMON_FEATURE_COLUMNS.")
