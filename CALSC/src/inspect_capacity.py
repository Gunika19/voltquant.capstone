from pathlib import Path
import pandas as pd

f = Path("data/01_raw/calce/CS2_35/CS2_35/CS2_35_8_30_10.xlsx")
xl = pd.ExcelFile(f)
df = pd.read_excel(xl, sheet_name="Channel_1-008")

for cyc in [1, 2, 3, 5, 10, 20, 50]:
    sub = df[df["Cycle_Index"] == cyc]
    dis_sub = sub[sub["Step_Index"] == 7]
    if not dis_sub.empty:
        q_start = dis_sub["Discharge_Capacity(Ah)"].iloc[0]
        q_end = dis_sub["Discharge_Capacity(Ah)"].iloc[-1]
        delta_q = q_end - q_start
        # Also compute current integration: trapz of current over test time
        dt = dis_sub["Test_Time(s)"].diff().fillna(0)
        q_int = -(dis_sub["Current(A)"] * dt).sum() / 3600.0
        print(f"Cycle {cyc}: Q_start={q_start:.4f}, Q_end={q_end:.4f}, Delta_Q={delta_q:.4f}, Integrated_Q={q_int:.4f}")
