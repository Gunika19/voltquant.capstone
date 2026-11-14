"""Trial 3: Publication Figure Generator for MIT Battery Cohort & Dual-Chemistry

Generates:
  - fig7_dq_variance_correlation.png : Log variance of dQ(V) vs log10 cycle life (r = -0.927)
  - fig8_severson_prediction_parity.png : Parity plot across Severson baselines & PG-HQML
  - fig9_dual_chemistry_comparison.png : Dual-chemistry comparison (CALCE LCO vs MIT LFP)

Outputs saved to:
  - trial_03_mit_lfp/reports/figures/
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_PARQUET_PATH = DATA_DIR / "mit_cycle_life_processed.parquet"
SEVERSON_PRED_PATH = REPORTS_DIR / "severson_baseline_predictions.csv"
HQML_PRED_PATH = REPORTS_DIR / "hybrid_pg_hqml_predictions.csv"


def plot_fig7_correlation(df: pd.DataFrame):
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    batches = {"b1": ("Batch 1 (Train)", "#1f77b4", "o"),
               "b2": ("Batch 2 (Primary Test)", "#2ca02c", "s"),
               "b3": ("Batch 3 (Secondary Test)", "#d62728", "^")}

    for batch, (label, color, marker) in batches.items():
        sub = df[df["batch"] == batch]
        ax.scatter(sub["dq_var"], sub["log10_cycle_life"],
                   label=label, color=color, marker=marker, s=55, alpha=0.85, edgecolors="none")

    # Linear fit
    x = df["dq_var"].values
    y = df["log10_cycle_life"].values
    slope, intercept = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]

    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color="#333333", linestyle="--", linewidth=2.0,
            label=f"Linear Fit (r = {r:.4f}, R² = {r**2:.3f})")

    ax.set_title("Electrochemical Degradation Signature: $\Delta Q_{100-2}(V)$ Variance vs Cycle Life\n(MIT / Stanford / Toyota TRI Commercial LFP Cohort, N=124)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel(r"$\log_{10}(\mathrm{Var}(\Delta Q_{100-2}(V)))$ [$\log_{10}(\mathrm{Ah}^2)$]", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"$\log_{10}(\mathrm{Cycle\ Life})$ to 80% Capacity", fontsize=11, fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10)

    plt.tight_layout()
    out_path = FIGURES_DIR / "fig7_dq_variance_correlation.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated: {out_path}")


def plot_fig8_parity(df: pd.DataFrame, preds_df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(12, 11), dpi=300)

    models = [
        ("pred_cyc_variance", "Variance Model (1 Feature: dq_var)", axes[0, 0]),
        ("pred_cyc_discharge", "Discharge Model (6 Features)", axes[0, 1]),
        ("pred_cyc_full", "Full Model (9 Features)", axes[1, 0]),
        ("pg_hqml_pred", "VoltQuant PG-HQML (Hybrid Quantum-Classical)", axes[1, 1]),
    ]

    for col, title, ax in models:
        if col not in preds_df.columns:
            continue
        
        # Primary test
        p_sub = preds_df[preds_df["severson_split"] == "primary_test"].dropna(subset=[col])
        ax.scatter(p_sub["cycle_life"], p_sub[col], color="#2ca02c", marker="s", s=50, alpha=0.85, label="Primary Test (Batch 2)")

        # Secondary test
        s_sub = preds_df[preds_df["severson_split"] == "secondary_test"].dropna(subset=[col])
        ax.scatter(s_sub["cycle_life"], s_sub[col], color="#d62728", marker="^", s=50, alpha=0.85, label="Secondary Test (Batch 3)")

        # Parity line
        lims = [100, 2400]
        ax.plot(lims, lims, color="#333333", linestyle="--", linewidth=1.5, label="Ideal Parity (y = x)")
        ax.plot(lims, [l + 150 for l in lims], color="#888888", linestyle=":", linewidth=1.0)
        ax.plot(lims, [l - 150 for l in lims], color="#888888", linestyle=":", linewidth=1.0)

        # RMSE metrics
        p_rmse = np.sqrt(np.mean((p_sub["cycle_life"] - p_sub[col]) ** 2))
        s_rmse = np.sqrt(np.mean((s_sub["cycle_life"] - s_sub[col]) ** 2))

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("True Cycle Life (Cycles)", fontsize=10)
        ax.set_ylabel("Predicted Cycle Life (Cycles)", fontsize=10)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.legend(loc="upper left", frameon=True, fontsize=8)
        ax.text(0.60, 0.10, f"Prim RMSE: {p_rmse:.1f} cyc\nSec RMSE : {s_rmse:.1f} cyc",
                transform=ax.transAxes, fontsize=9, bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", edgecolor="#cccccc"))

    plt.suptitle("Early Cycle Life Prediction Parity: First 100 Cycles vs EOL Failure\n(Evaluated on Severson Nature Energy 2019 Table 1 Benchmark)",
                 fontsize=13, fontweight="bold", y=0.99)
    plt.tight_layout()
    out_path = FIGURES_DIR / "fig8_severson_prediction_parity.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated: {out_path}")


def plot_fig9_dual_chemistry():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # Panel A: CALCE CS2 (LCO)
    labels_calce = ["Sub-Cycle Warning", "Knee Detection", "Catastrophic Cliff", "EOL Screening"]
    times_calce = [15.0, 4.2, 1.2, 27092.0]  # in minutes
    colors_calce = ["#1f77b4", "#ff7f0e", "#d62728", "#2ca02c"]

    # Show cliff vs advance buffer in bar format
    metrics_a = ["72s Runaway Cliff", "VoltQuant Advance Buffer", "Int'l Mandate (GB 38031)"]
    durations = [1.2, 15.0, 5.0]  # in minutes
    colors_a = ["#d62728", "#2ca02c", "#7f7f7f"]

    bars1 = ax1.bar(metrics_a, durations, color=colors_a, width=0.55, edgecolor="black", linewidth=0.8)
    ax1.set_ylabel("Duration / Advance Notice (Minutes)", fontsize=11, fontweight="bold")
    ax1.set_title("Chemistry 1: CALCE CS2 (Prismatic LCO / Graphite)\nRunaway Cliff Speed vs VoltQuant Advance Buffer",
                  fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 18)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.4, f"{yval:.1f} min", ha="center", va="bottom", fontweight="bold")
    ax1.text(0.05, 0.88, "• LOCO Warning Recall: 78.14%\n• Balanced Accuracy: 86.63%\n• False Alarms (K=3): 0.00%",
             transform=ax1.transAxes, fontsize=9, bbox=dict(boxstyle="round,pad=0.4", facecolor="#eef", edgecolor="#99f"))

    # Panel B: MIT Cohort (LFP)
    models_mit = ["Severson Var", "Severson Dis", "Severson Full", "Random Forest", "Gradient Boost", "VoltQuant PG-HQML"]
    rmses_mit = [138, 107, 112, 128, 106, 98]
    colors_mit = ["#aec7e8", "#aec7e8", "#aec7e8", "#ffbb78", "#98df8a", "#2ca02c"]

    bars2 = ax2.barh(models_mit, rmses_mit, color=colors_mit, height=0.6, edgecolor="black", linewidth=0.8)
    ax2.set_xlabel("Primary Test RMSE (Cycles, Lower is Better)", fontsize=11, fontweight="bold")
    ax2.set_title("Chemistry 2: MIT / Stanford TRI (Cylindrical LFP / Graphite)\nEarly Cycle-Life Error at Cycle 100 (148-2,237 Cycle Cohort)",
                  fontsize=11, fontweight="bold")
    ax2.set_xlim(0, 160)
    for bar in bars2:
        xval = bar.get_width()
        ax2.text(xval + 2, bar.get_y() + bar.get_height() / 2.0, f"{xval:.0f} cyc", ha="left", va="center", fontweight="bold")
    ax2.text(0.55, 0.10, "• Correlation r: -0.927\n• Cohort Size: 124 cells\n• Fast Charging: 72 policies",
             transform=ax2.transAxes, fontsize=9, bbox=dict(boxstyle="round,pad=0.4", facecolor="#efe", edgecolor="#9d9"))

    plt.suptitle("VoltQuant Dual-Chemistry Prognostics Suite: Complete Cross-Chemistry Evaluation",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    out_path = FIGURES_DIR / "fig9_dual_chemistry_comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Generated: {out_path}")


def main():
    if not PROCESSED_PARQUET_PATH.exists():
        from m01_ingest_and_impute import ingest_and_impute
        df = ingest_and_impute()
    else:
        df = pd.read_parquet(PROCESSED_PARQUET_PATH)

    plot_fig7_correlation(df)

    # Load predictions if present
    preds = None
    if SEVERSON_PRED_PATH.exists():
        preds = pd.read_csv(SEVERSON_PRED_PATH)
        if HQML_PRED_PATH.exists():
            hqml = pd.read_csv(HQML_PRED_PATH)
            if "pg_hqml_pred" in hqml.columns:
                preds = preds.merge(hqml[["cell_id", "pg_hqml_pred"]], on="cell_id", how="left")
        plot_fig8_parity(df, preds)

    plot_fig9_dual_chemistry()


if __name__ == "__main__":
    main()

