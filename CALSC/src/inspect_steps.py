from pathlib import Path
import pandas as pd

f = Path("data/01_raw/calce/CS2_35/CS2_35/CS2_35_8_30_10.xlsx")
xl = pd.ExcelFile(f)
df = pd.read_excel(xl, sheet_name="Channel_1-008")

step_stats = df.groupby("Step_Index").agg(
    count=("Current(A)", "count"),
    mean_I=("Current(A)", "mean"),
    min_I=("Current(A)", "min"),
    max_I=("Current(A)", "max"),
    min_V=("Voltage(V)", "min"),
    max_V=("Voltage(V)", "max"),
    max_Q_dis=("Discharge_Capacity(Ah)", "max"),
    max_Q_ch=("Charge_Capacity(Ah)", "max")
)
print("=== STEP INDEX SUMMARY ===")
print(step_stats)

cyc1 = df[df["Cycle_Index"] == 1]
print("\nCycle 1 Step sequence:", cyc1["Step_Index"].unique())
for s in cyc1["Step_Index"].unique():
    sub = cyc1[cyc1["Step_Index"] == s]
    print(f"  Step {s}: len={len(sub)}, I=[{sub['Current(A)'].min():.3f}, {sub['Current(A)'].max():.3f}], V=[{sub['Voltage(V)'].min():.3f}, {sub['Voltage(V)'].max():.3f}], Q_dis_max={sub['Discharge_Capacity(Ah)'].max():.4f}")
