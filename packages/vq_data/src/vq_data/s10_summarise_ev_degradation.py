"""Descriptive summary of ev_battery_degradation_v1.csv -- documentation, not a pipeline stage.

Reads data/01_raw/ev_degradation/ev_battery_degradation_v1.csv, writes
reports/ev_degradation_summary.md. Writes no cleaned/intermediate data file
-- there is nothing to clean this source into: per CLAUDE.md, it has 4
positives in 10,000 rows (0.04%), which cannot support a supervised claim.
Its role is descriptive context for the paper, not training data.
"""

from __future__ import annotations

import pandas as pd

from vq_core.paths import PROJECT_ROOT, RAW_EV_DEGRADATION

OUTPUT_PATH = PROJECT_ROOT / "reports" / "ev_degradation_summary.md"

COLUMN_DESCRIPTIONS = {
    "Vehicle_ID": "Opaque per-vehicle identifier (hex string).",
    "Car_Model": "Vehicle make/model (5 distinct values).",
    "Battery_Type": "Cell chemistry: NMC or LFP.",
    "Battery_Capacity_kWh": "Nameplate pack capacity, kWh.",
    "Vehicle_Age_Months": "Vehicle age, months.",
    "Total_Charging_Cycles": "Cumulative charge cycles to date.",
    "Avg_Temperature_C": "Average operating temperature, degrees C.",
    "Fast_Charge_Ratio": "Fraction of charges that were fast-charges (0-1).",
    "Avg_Discharge_Rate_C": "Average discharge rate, C-rate units.",
    "Driving_Style": "Categorical: Aggressive, Moderate, or Conservative.",
    "Internal_Resistance_Ohm": "Internal resistance, ohms.",
    "SoH_Percent": "State of health, percent (100 = new).",
    "Battery_Status": "Categorical label: Healthy or Replace Required.",
}


def section(title: str) -> None:
    print()
    print(f"=== {title} ===")


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    section("Load")
    df = pd.read_csv(RAW_EV_DEGRADATION)
    print(f"shape: {df.shape}")

    section("Battery_Status counts")
    status_counts = df["Battery_Status"].value_counts()
    print(status_counts.to_string())

    section("SoH distribution by status")
    soh_by_status = df.groupby("Battery_Status")["SoH_Percent"].describe()
    print(soh_by_status.to_string())

    n_positive = int(status_counts.get("Replace Required", 0))
    n_total = len(df)

    lines = []
    lines.append("# ev_battery_degradation_v1.csv: descriptive summary")
    lines.append("")
    lines.append(
        "Documentation for the paper, not a pipeline stage -- this source is not cleaned into a "
        "feature table because it cannot support a supervised claim (see the note at the end)."
    )
    lines.append("")

    lines.append("## Shape")
    lines.append("")
    lines.append(f"{df.shape[0]} rows x {df.shape[1]} columns.")
    lines.append("")

    lines.append("## Columns")
    lines.append("")
    lines.append("| column | description |")
    lines.append("|---|---|")
    for column in df.columns:
        lines.append(f"| `{column}` | {COLUMN_DESCRIPTIONS.get(column, '(undocumented)')} |")
    lines.append("")

    lines.append("## Battery_Status counts")
    lines.append("")
    lines.append("| status | count | pct |")
    lines.append("|---|---|---|")
    for status, count in status_counts.items():
        lines.append(f"| {status} | {count} | {100 * count / n_total:.2f}% |")
    lines.append("")

    lines.append("## SoH_Percent distribution by Battery_Status")
    lines.append("")
    lines.append("| status | count | mean | std | min | 25% | 50% | 75% | max |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for status, row in soh_by_status.iterrows():
        lines.append(
            f"| {status} | {row['count']:.0f} | {row['mean']:.2f} | {row['std']:.2f} | {row['min']:.2f} | "
            f"{row['25%']:.2f} | {row['50%']:.2f} | {row['75%']:.2f} | {row['max']:.2f} |"
        )
    lines.append("")

    lines.append("## Note")
    lines.append("")
    lines.append(
        f"With {n_positive} positive(s) (`Replace Required`) in {n_total} rows "
        f"({100 * n_positive / n_total:.2f}%), this source is documented context for the paper, not "
        f"training data -- {n_positive} examples cannot support any supervised claim, regardless of how "
        f"separable they look in `SoH_Percent` (see the distribution above)."
    )
    lines.append("")

    OUTPUT_PATH.write_text("\n".join(lines))

    section("Wrote")
    print(f"{OUTPUT_PATH} ({n_total} rows summarised, no cleaned data file written)")


if __name__ == "__main__":
    main()
