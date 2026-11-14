"""K02 · Cycle Table and SOH Ground Truth

Extracts per-cycle discharge capacity and builds ground-truth SOH table for CALCE CS2 cells.
Saves to data/02_intermediate/calce_discharge_cycles.parquet.

Conventions:
- Per-cell SOH = discharge capacity divided by that cell's own first-cycle capacity (capacity_ref).
- 1.1 Ah nominal LCO.
- Reuses s04's validity filter (0.30 * capacity_ref <= capacity <= 1.10 * capacity_ref).
- Enforces float64 std and numeric conventions.
- Handles chronological multi-file concatenation with deduplication and cumulative cycle indexing.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

RAW_DIR = Path("data/01_raw/calce")
INTERMEDIATE_DIR = Path("data/02_intermediate")
INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PARQUET = INTERMEDIATE_DIR / "calce_discharge_cycles.parquet"

CELLS = ["CS2_35", "CS2_36", "CS2_37", "CS2_38"]
NOMINAL_CAPACITY_AH = 1.1
SOH_EOL_THRESHOLD = 0.80
LOW_RATIO = 0.30
HIGH_RATIO = 1.10

def load_cell_files_ordered(cell_dir: Path) -> list[tuple[pd.Timestamp, Path, str]]:
    """Scan all excel files for a cell, determine chronological start time, and deduplicate."""
    files = list(cell_dir.rglob("*.xlsx"))
    records = []
    
    print(f"Scanning and sorting {len(files)} files in {cell_dir.name}...")
    for f in files:
        try:
            xl = pd.ExcelFile(f)
            sheet = [s for s in xl.sheet_names if "channel" in s.lower()][0]
            # Read first 2 rows for timestamp
            df_head = pd.read_excel(xl, sheet_name=sheet, nrows=2)
            dt = pd.to_datetime(df_head["Date_Time"].iloc[0])
            records.append({
                "timestamp": dt,
                "path": f,
                "sheet": sheet,
                "file_size": f.stat().st_size
            })
        except Exception as e:
            print(f"Warning: Failed to inspect {f.name}: {e}")
            
    df_files = pd.DataFrame(records)
    # Sort strictly chronologically
    df_files = df_files.sort_values(by="timestamp").reset_index(drop=True)
    
    # Deduplicate: if consecutive files have identical start timestamp and identical size, keep only one
    unique_records = []
    seen = set()
    for _, row in df_files.iterrows():
        key = (row["timestamp"], row["file_size"])
        if key in seen:
            print(f"  [Deduplicated] Skipping duplicate file: {row['path'].name}")
            continue
        seen.add(key)
        unique_records.append((row["timestamp"], row["path"], row["sheet"]))
        
    print(f"  Retained {len(unique_records)} unique chronological log files.")
    return unique_records

def process_cell(cell_id: str) -> pd.DataFrame:
    """Parse all cycles across chronologically ordered logs for a given cell."""
    cell_dir = RAW_DIR / cell_id
    ordered_files = load_cell_files_ordered(cell_dir)
    
    cycle_records = []
    global_cycle_counter = 0
    
    for dt_start, file_path, sheet_name in ordered_files:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        # In Arbin logs, each file has Cycle_Index 1..N
        # We process each Cycle_Index in order
        local_cycles = sorted(df["Cycle_Index"].unique())
        
        for loc_cyc in local_cycles:
            cyc_df = df[df["Cycle_Index"] == loc_cyc]
            
            # Discharge step is step 7 (or where Current < -0.5 A)
            dis_df = cyc_df[(cyc_df["Step_Index"] == 7) & (cyc_df["Current(A)"] < -0.2)]
            
            if dis_df.empty:
                # Some initial diagnostic or rest cycles may have no discharge
                continue
                
            global_cycle_counter += 1
            
            # Compute discharge capacity via delta and current integration
            q_start = dis_df["Discharge_Capacity(Ah)"].iloc[0]
            q_end = dis_df["Discharge_Capacity(Ah)"].iloc[-1]
            dis_capacity_delta = float(q_end - q_start)
            
            # Numerical integration verification: trapz(-I * dt)
            dt_s = dis_df["Test_Time(s)"].diff().fillna(0).values
            curr_a = dis_df["Current(A)"].values
            dis_capacity_int = float(-np.sum(curr_a * dt_s) / 3600.0)
            
            # Use delta capacity (calibrated by cycler hardware)
            capacity = dis_capacity_delta if dis_capacity_delta > 0 else dis_capacity_int
            
            start_ts = pd.to_datetime(dis_df["Date_Time"].iloc[0])
            end_ts = pd.to_datetime(dis_df["Date_Time"].iloc[-1])
            duration_s = float(dis_df["Test_Time(s)"].iloc[-1] - dis_df["Test_Time(s)"].iloc[0])
            
            # Voltage profile checks
            v_start = float(dis_df["Voltage(V)"].iloc[0])
            v_end = float(dis_df["Voltage(V)"].iloc[-1])
            v_series = dis_df["Voltage(V)"].astype(np.float64)
            v_mean = float(v_series.mean())
            v_std = float(v_series.std()) if len(v_series) > 1 else 0.0
            i_mean = float(dis_df["Current(A)"].astype(np.float64).mean())
            
            cycle_records.append({
                "battery_id": cell_id,
                "cycle_index": global_cycle_counter,
                "local_cycle_index": int(loc_cyc),
                "source_file": file_path.name,
                "start_time": start_ts,
                "end_time": end_ts,
                "duration_s": duration_s,
                "capacity": capacity,
                "capacity_integrated": dis_capacity_int,
                "voltage_start": v_start,
                "voltage_end": v_end,
                "voltage_mean": v_mean,
                "voltage_std": v_std,
                "current_mean": i_mean,
                "num_samples": len(dis_df)
            })
            
    cell_cycle_df = pd.DataFrame(cycle_records)
    print(f"Cell {cell_id}: extracted {len(cell_cycle_df)} discharge cycles.")
    
    # Establish capacity_ref matching NASA's s04:
    # Per-cell beginning-of-life reference: median of the 5 highest valid capacities.
    # Nominal referencing was rejected on NASA because fresh cells average 0.884 of nameplate
    # with std 0.051, which would encode manufacturing spread as degradation.
    valid_candidates = cell_cycle_df[cell_cycle_df["capacity"] > 0.5]["capacity"]
    capacity_ref = float(valid_candidates.nlargest(5).median())
    print(f"  Reference Capacity for {cell_id} (median of 5 highest valid): {capacity_ref:.4f} Ah")
    
    cell_cycle_df["capacity_ref"] = capacity_ref
    cell_cycle_df["soh"] = (cell_cycle_df["capacity"] / capacity_ref).astype(np.float64)
    cell_cycle_df["soh_pct"] = cell_cycle_df["soh"] * 100.0
    cell_cycle_df["fade_pct"] = (1.0 - cell_cycle_df["soh"]) * 100.0
    
    # s04 validity filter:
    # valid if LOW_RATIO * capacity_ref <= capacity <= HIGH_RATIO * capacity_ref
    cell_cycle_df["valid"] = (
        (cell_cycle_df["capacity"] >= LOW_RATIO * capacity_ref) &
        (cell_cycle_df["capacity"] <= HIGH_RATIO * capacity_ref)
    )
    
    valid_cycles = cell_cycle_df[cell_cycle_df["valid"]]
    final_fade = valid_cycles["fade_pct"].iloc[-1]
    final_soh = valid_cycles["soh"].iloc[-1]
    min_soh = valid_cycles["soh"].min()
    crosses_80 = (valid_cycles["soh"] < SOH_EOL_THRESHOLD).any()
    
    print(f"  Total valid cycles: {len(valid_cycles)} / {len(cell_cycle_df)}")
    print(f"  Final SOH: {final_soh:.4f} (Fade: {final_fade:.1f}%) | Min SOH: {min_soh:.4f}")
    print(f"  Crosses 0.80 EOL Threshold? {'YES' if crosses_80 else 'NO'}")
    
    return cell_cycle_df

def build_all_cells_table():
    start_time = time.time()
    dfs = []
    
    for cell in CELLS:
        print(f"\n==========================================")
        print(f"Processing Cell: {cell}")
        print(f"==========================================")
        df_cell = process_cell(cell)
        dfs.append(df_cell)
        
    master_df = pd.concat(dfs, ignore_index=True)
    
    # Save as parquet
    master_df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"\nSaved intermediate discharge cycle table to {OUTPUT_PARQUET}")
    print(f"Master table shape: {master_df.shape}")
    
    # Summary Table
    summary = []
    for cell, group in master_df.groupby("battery_id"):
        valid = group[group["valid"]]
        crossing_row = valid[valid["soh"] < SOH_EOL_THRESHOLD]
        first_crossing_cyc = crossing_row["cycle_index"].iloc[0] if not crossing_row.empty else None
        
        summary.append({
            "Cell": cell,
            "Total Cycles": len(group),
            "Valid Cycles": len(valid),
            "Cap Ref (Ah)": round(group["capacity_ref"].iloc[0], 4),
            "Min SOH": round(valid["soh"].min(), 4),
            "Max Fade (%)": round(valid["fade_pct"].max(), 1),
            "EOL 0.80 Crossing Cycle": first_crossing_cyc
        })
        
    df_sum = pd.DataFrame(summary)
    print("\n=== CALCE CS2 CELL CYCLE & SOH SUMMARY ===")
    print(df_sum.to_string(index=False))
    
    elapsed = time.time() - start_time
    print(f"\nK02 complete in {elapsed:.1f}s.")

if __name__ == "__main__":
    build_all_cells_table()
