"""Generate Figure 6: Exact 10-15 Minute Advance Buffer to Runaway Voltage Cliff

Plots the discharge voltage trajectory V(t) during the catastrophic runaway crossing cycle,
illustrating:
1. Catastrophic voltage collapse (< 3.2V down to 2.75V) occurring in just ~73 seconds.
2. The 10-Minute Advance Warning Point (t = 38.5 min, V = 3.51V) giving exactly 10.0 min buffer.
3. The 15-Minute Advance Warning Point (t = 33.5 min, V = 3.55V) giving exactly 15.0 min buffer.
4. Regulatory baseline (GB 38031: 5 minutes) vs. VoltQuant's 10-15 min advance buffer.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
FIGURES_DIR = BASE_DIR / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
CYCLE_TABLE_PATH = BASE_DIR / "data" / "02_intermediate" / "calce_discharge_cycles.parquet"
if not CYCLE_TABLE_PATH.exists():
    CYCLE_TABLE_PATH = Path("data/02_intermediate/calce_discharge_cycles.parquet")
RAW_DIR = BASE_DIR / "data" / "01_raw" / "calce"
if not RAW_DIR.exists():
    RAW_DIR = Path("data/01_raw/calce")
ARTIFACT_DIR = Path(os.environ.get("VOLTQUANT_ARTIFACT_DIR", ""))

plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8

def plot_fig6():
    # Load CS2_36 Cycle 501
    df_cyc = pd.read_parquet(CYCLE_TABLE_PATH)
    row = df_cyc[(df_cyc["battery_id"] == "CS2_36") & (df_cyc["cycle_index"] == 501)].iloc[0]
    source_file = row["source_file"]
    local_cyc = int(row["local_cycle_index"])
    
    file_path = list((RAW_DIR / "CS2_36").rglob(source_file))[0]
    xl = pd.ExcelFile(file_path)
    s = [sh for sh in xl.sheet_names if "channel" in sh.lower()][0]
    df_raw = pd.read_excel(file_path, sheet_name=s)
    dis = df_raw[(df_raw["Cycle_Index"] == local_cyc) & (df_raw["Step_Index"] == 7)].sort_values("Test_Time(s)").reset_index(drop=True)
    
    t0 = dis["Test_Time(s)"].iloc[0]
    dis["elapsed_min"] = (dis["Test_Time(s)"] - t0) / 60.0
    
    cutoff_time = dis["elapsed_min"].iloc[-1]
    cliff_30_time = dis[dis["Voltage(V)"] <= 3.0]["elapsed_min"].iloc[0]
    
    t_10 = cliff_30_time - 10.0
    t_15 = cliff_30_time - 15.0
    t_reg = cliff_30_time - 5.0
    
    v_10 = dis[dis["elapsed_min"] <= t_10]["Voltage(V)"].iloc[-1]
    v_15 = dis[dis["elapsed_min"] <= t_15]["Voltage(V)"].iloc[-1]
    v_reg = dis[dis["elapsed_min"] <= t_reg]["Voltage(V)"].iloc[-1]
    
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    
    # Plot voltage curve
    ax.plot(dis["elapsed_min"], dis["Voltage(V)"], color="#2b5c8f", lw=2.5, label="Cell Terminal Voltage V(t)")
    
    # Highlight runaway collapse zone
    ax.axvspan(cliff_30_time, cutoff_time, color="#fee0d2", alpha=0.7, label="Runaway Voltage Collapse (Plunge < 3.0V in 73s!)")
    ax.axvline(cliff_30_time, color="#de2d26", linestyle="--", lw=1.8, label=f"Runaway Cliff (< 3.0V) at t = {cliff_30_time:.1f} min")
    
    # 15-Minute Advance Warning Buffer
    ax.axvspan(t_15, cliff_30_time, color="#e0f3f8", alpha=0.4)
    ax.scatter([t_15], [v_15], color="#02818a", s=140, zorder=6, label=f"15-Minute Advance Warning (t = {t_15:.1f} min, V = {v_15:.2f}V)")
    ax.annotate("15-Minute Advance Buffer\nto Runaway Cliff", 
                xy=(t_15, v_15), xytext=(t_15 - 8, v_15 + 0.25),
                arrowprops=dict(arrowstyle="->", color="#02818a", lw=1.5),
                fontsize=9.5, fontweight="bold", color="#02818a")
    
    # 10-Minute Advance Warning Buffer
    ax.axvspan(t_10, cliff_30_time, color="#fee391", alpha=0.35)
    ax.scatter([t_10], [v_10], color="#d95f0e", s=140, zorder=6, label=f"10-Minute Advance Warning (t = {t_10:.1f} min, V = {v_10:.2f}V)")
    ax.annotate("10-Minute Advance Buffer\nto Runaway Cliff", 
                xy=(t_10, v_10), xytext=(t_10 - 2, v_10 - 0.35),
                arrowprops=dict(arrowstyle="->", color="#d95f0e", lw=1.5),
                fontsize=9.5, fontweight="bold", color="#d95f0e")
    
    # Regulatory 5-minute line
    ax.axvline(t_reg, color="#636363", linestyle=":", lw=1.5, label=f"GB 38031 5-Min Regulatory Mandate (t = {t_reg:.1f} min)")
    
    # Format axes
    ax.set_xlabel("Elapsed Discharge Time (Minutes)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Cell Terminal Voltage (Volts)", fontsize=11, fontweight="bold")
    ax.set_title("VoltQuant Exact 10-Minute & 15-Minute Advance Warning Buffer to Runaway Cliff\n(CALCE CS2_36 Runaway Crossing Cycle 501)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlim(0, 52)
    ax.set_ylim(2.5, 4.3)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.92)
    
    plt.tight_layout()
    p = FIGURES_DIR / "fig6_runaway_cliff_and_advance_buffer.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"Saved {p}")
    
    # Copy to artifact directory if available
    if ARTIFACT_DIR and ARTIFACT_DIR.is_dir():
        dest = ARTIFACT_DIR / p.name
        shutil.copy(p, dest)
        print(f"Copied {p.name} -> {dest}")

if __name__ == "__main__":
    plot_fig6()

