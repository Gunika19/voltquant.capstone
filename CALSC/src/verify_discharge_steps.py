from pathlib import Path
import pandas as pd

raw_dir = Path("data/01_raw/calce")

for cell in ["CS2_35", "CS2_36", "CS2_37", "CS2_38"]:
    files = list((raw_dir / cell).rglob("*.xlsx"))
    print(f"\nChecking {cell} across {len(files)} files...")
    
    # Check 3 random/spread files
    sample_indices = [0, len(files)//2, len(files)-1]
    for idx in sample_indices:
        f = files[idx]
        xl = pd.ExcelFile(f)
        sheet = [s for s in xl.sheet_names if "channel" in s.lower()][0]
        df = pd.read_excel(xl, sheet_name=sheet, nrows=5000)
        
        # Check negative current steps
        neg_steps = df[df["Current(A)"] < -0.5]["Step_Index"].unique()
        print(f"  {f.name}: negative current steps = {neg_steps}")
