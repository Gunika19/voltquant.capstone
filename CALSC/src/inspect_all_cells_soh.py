"""Inspect early cycles and discharge capacity behavior across all 4 cells."""

from pathlib import Path
import pandas as pd
import numpy as np

raw_dir = Path("data/01_raw/calce")
cells = ["CS2_35", "CS2_36", "CS2_37", "CS2_38"]

for cell in cells:
    files = list((raw_dir / cell).rglob("*.xlsx"))
    print(f"\n==================== {cell} ({len(files)} files) ====================")
    
    file_records = []
    for f in files:
        xl = pd.ExcelFile(f)
        sheet = [s for s in xl.sheet_names if "channel" in s.lower()][0]
        # read first row for date
        df_head = pd.read_excel(xl, sheet_name=sheet, nrows=2)
        dt = pd.to_datetime(df_head["Date_Time"].iloc[0])
        file_records.append((dt, f, sheet))
        
    file_records.sort(key=lambda x: x[0])
    
    # Check first 5 files
    print("Early files:")
    for dt, f, sheet in file_records[:5]:
        df = pd.read_excel(f, sheet_name=sheet)
        dis_steps = df[df["Step_Index"] == 7]
        cycs = df["Cycle_Index"].nunique()
        max_q = dis_steps["Discharge_Capacity(Ah)"].max() if not dis_steps.empty else 0
        print(f"  {dt} | {f.name} | rows={len(df)} | cycles={cycs} | dis_rows={len(dis_steps)} | max_Q_dis={max_q:.4f}")
