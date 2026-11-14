import pandas as pd
import numpy as np

df = pd.read_parquet("data/02_intermediate/calce_discharge_cycles.parquet")

def get_sustained_crossing(series, threshold=0.80, persistence=3):
    below = (series < threshold).values
    for i in range(len(below) - persistence + 1):
        if np.all(below[i : i + persistence]):
            return i
    return None

for cell, group in df.groupby("battery_id"):
    valid = group[group["valid"]].sort_values("cycle_index").reset_index(drop=True)
    cross_idx = get_sustained_crossing(valid["soh"], threshold=0.80, persistence=3)
    if cross_idx is not None:
        cross_cyc = valid.loc[cross_idx, "cycle_index"]
        # Fixed 20 window
        fixed_w = 20
        # Proportional window (~12% of pre-crossing cycles)
        prop_w = max(5, int(round(0.12 * cross_cyc)))
        
        print(f"\n{cell}:")
        print(f"  Total valid cycles: {len(valid)}")
        print(f"  Sustained crossing at cycle_index: {cross_cyc} (index {cross_idx})")
        print(f"  Fixed 20 Warning window: cycles {cross_cyc - fixed_w} to {cross_cyc - 1} (len {fixed_w})")
        print(f"  Proportional Warning window (12%): cycles {cross_cyc - prop_w} to {cross_cyc - 1} (len {prop_w})")
        print(f"  Runaway/Failed: cycles >= {cross_cyc} ({len(valid) - cross_idx} cycles)")
