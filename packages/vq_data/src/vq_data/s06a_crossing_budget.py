"""Determine what lead times are feasible before defining early-warning labels.

Read-only diagnostic on data/02_intermediate/nasa_discharge_cycles.parquet
and data/03_features/nasa_discharge_measurements.parquet. Prints a report
to stdout and writes a plot to reports/figures/. Writes no data files.

Labelling scheme evaluated here (not yet applied anywhere):
  RUNAWAY : discharge_cycle_index >= crossing_index (soh has crossed below
            the threshold)
  WARNING : crossing_index - lead_time <= discharge_cycle_index < crossing_index
  NORMAL  : discharge_cycle_index < crossing_index - lead_time
Only cells whose soh actually crosses the threshold have a crossing_index;
cells that never cross are excluded from the lead-time tables (there is no
anchor to count backwards from) but are named in the crossing inventory.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from vq_core.paths import PROJECT_ROOT

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
LABELS_PATH = PROJECT_ROOT / "data" / "02_intermediate" / "nasa_discharge_cycles.parquet"
MEASUREMENTS_PATH = PROJECT_ROOT / "data" / "03_features" / "nasa_discharge_measurements.parquet"

SCOPE_VERDICTS = ["CLEAN", "STRATIFIED"]
DEFAULT_SOH_THRESHOLD = 0.80
SENSITIVITY_THRESHOLDS = [0.75, 0.80, 0.85]
LEAD_TIMES = [5, 10, 15, 20, 30, 50]
PLOT_LEAD_TIME = 20
MIN_WORKABLE_CROSSING_CELLS = 10


def section(title: str) -> None:
    print()
    print(f"=== {title} ===")


def load_scope() -> pd.DataFrame:
    section("Scope")
    labels = pd.read_parquet(LABELS_PATH)
    qualifying = labels[labels["verdict"].isin(SCOPE_VERDICTS) & (~labels["invalid_measurement"])].copy()

    stratified_cells = sorted(qualifying.loc[qualifying["verdict"] == "STRATIFIED", "battery_id"].unique())
    for battery_id in stratified_cells:
        battery_rows = qualifying[qualifying["battery_id"] == battery_id]
        strata_counts = battery_rows.groupby(["ambient_temperature", "load_level"]).size()
        dominant_temp, dominant_load = strata_counts.idxmax()
        dominant_count = strata_counts.max()
        outside_dominant = (qualifying["battery_id"] == battery_id) & ~(
            (qualifying["ambient_temperature"] == dominant_temp) & (qualifying["load_level"] == dominant_load)
        )
        print(
            f"{battery_id} is STRATIFIED -- restricting to its dominant stratum "
            f"(ambient_temperature={dominant_temp}, load_level={dominant_load}, {dominant_count} of "
            f"{len(battery_rows)} valid cycles); dropping {int(outside_dominant.sum())} cycle(s) from other strata"
        )
        qualifying = qualifying[~outside_dominant]

    cells = sorted(qualifying["battery_id"].unique())
    print(f"\ncells in scope: {len(cells)} -> {cells}")
    print(f"total valid discharge cycles in scope: {len(qualifying)}")

    return qualifying


def find_crossings(qualifying: pd.DataFrame, threshold: float) -> dict[str, int]:
    crossing_index = {}
    for battery_id, group in qualifying.groupby("battery_id"):
        group = group.sort_values("discharge_cycle_index")
        crossed = group.loc[group["soh"] < threshold, "discharge_cycle_index"]
        if not crossed.empty:
            crossing_index[battery_id] = int(crossed.iloc[0])
    return crossing_index


def crossing_inventory(qualifying: pd.DataFrame, threshold: float) -> dict[str, int]:
    section(f"Crossing inventory (soh < {threshold})")
    crossing_index = find_crossings(qualifying, threshold)

    rows = []
    for battery_id, group in qualifying.groupby("battery_id"):
        group = group.sort_values("discharge_cycle_index")
        crosses = battery_id in crossing_index
        ci = crossing_index.get(battery_id)
        n_before = int((group["discharge_cycle_index"] < ci).sum()) if crosses else len(group)
        n_after = int((group["discharge_cycle_index"] >= ci).sum()) if crosses else 0
        rows.append(
            {
                "battery_id": battery_id,
                "n_cycles": len(group),
                "min_soh": group["soh"].min(),
                "crosses": crosses,
                "crossing_index": ci,
                "n_before": n_before,
                "n_after": n_after,
            }
        )
    table = pd.DataFrame(rows).sort_values("battery_id").reset_index(drop=True)
    print(table.to_string(index=False))

    never_cross = table.loc[~table["crosses"], "battery_id"].tolist()
    print(f"\nnever cross {threshold}: {len(never_cross)} of {len(table)} ({100 * len(never_cross) / len(table):.1f}%) -> {never_cross}")

    artifact_cells = []
    for battery_id, ci in crossing_index.items():
        if ci != 0:
            continue
        group = qualifying[qualifying["battery_id"] == battery_id].sort_values("discharge_cycle_index")
        cycles_after_first = group[group["discharge_cycle_index"] > 0]
        if not cycles_after_first.empty and cycles_after_first["soh"].iloc[0] >= threshold:
            artifact_cells.append(battery_id)
    if artifact_cells:
        print(
            f"\n[NOTE] {len(artifact_cells)} crossing cell(s) cross at discharge_cycle_index=0 because "
            f"their very first valid cycle is a single noisy low reading that rebounds to >= {threshold} "
            f"on the next cycle, not sustained decline: {artifact_cells}. Reported literally as first "
            f"crossings below (as specified), but these contribute cycles/rows that are entirely RUNAWAY "
            f"in every lead-time table that follows -- see the recommendation at the end."
        )

    return crossing_index


def lead_time_feasibility(qualifying: pd.DataFrame, crossing_index: dict[str, int]) -> pd.DataFrame:
    section("Lead-time feasibility (cycle level, crossing cells only)")
    crossing_cells = qualifying[qualifying["battery_id"].isin(crossing_index)]

    rows = []
    for lead_time in LEAD_TIMES:
        warning = runaway = normal = 0
        insufficient = 0
        for battery_id, ci in crossing_index.items():
            group = crossing_cells[crossing_cells["battery_id"] == battery_id]
            warning_start = max(0, ci - lead_time)
            is_runaway = group["discharge_cycle_index"] >= ci
            is_warning = (~is_runaway) & (group["discharge_cycle_index"] >= warning_start)
            is_normal = ~is_runaway & ~is_warning
            warning += int(is_warning.sum())
            runaway += int(is_runaway.sum())
            normal += int(is_normal.sum())
            if ci < lead_time:
                insufficient += 1
        total = warning + runaway + normal
        rows.append(
            {
                "lead_time": lead_time,
                "normal": normal,
                "warning": warning,
                "runaway": runaway,
                "total": total,
                "normal_pct": 100 * normal / total,
                "warning_pct": 100 * warning / total,
                "runaway_pct": 100 * runaway / total,
                "cells_with_insufficient_history": insufficient,
            }
        )
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    return table


def measurement_level_feasibility(qualifying: pd.DataFrame, crossing_index: dict[str, int]) -> pd.DataFrame:
    section("Lead-time feasibility (measurement level, joined against nasa_discharge_measurements.parquet)")
    measurements = pd.read_parquet(MEASUREMENTS_PATH)
    # Semi-join on (battery_id, discharge_cycle_index) against `qualifying`,
    # which already carries only the dominant stratum's cycles for any
    # STRATIFIED cell -- this restricts measurement rows the same way,
    # without needing to special-case which cells are stratified.
    scope_cycles = qualifying.loc[qualifying["battery_id"].isin(crossing_index), ["battery_id", "discharge_cycle_index"]].drop_duplicates()
    scope = measurements.merge(scope_cycles, on=["battery_id", "discharge_cycle_index"], how="inner")

    scope["crossing_index"] = scope["battery_id"].map(crossing_index)
    print(f"measurement rows in scope (crossing cells, dominant stratum where applicable): {len(scope)}")

    rows = []
    for lead_time in LEAD_TIMES:
        warning_start = (scope["crossing_index"] - lead_time).clip(lower=0)
        is_runaway = scope["discharge_cycle_index"] >= scope["crossing_index"]
        is_warning = (~is_runaway) & (scope["discharge_cycle_index"] >= warning_start)
        is_normal = ~is_runaway & ~is_warning
        total = len(scope)
        rows.append(
            {
                "lead_time": lead_time,
                "normal": int(is_normal.sum()),
                "warning": int(is_warning.sum()),
                "runaway": int(is_runaway.sum()),
                "total": total,
                "normal_pct": 100 * is_normal.sum() / total,
                "warning_pct": 100 * is_warning.sum() / total,
                "runaway_pct": 100 * is_runaway.sum() / total,
            }
        )
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    return table


def threshold_sensitivity(qualifying: pd.DataFrame) -> float:
    section("Sensitivity: crossing-cell count across SOH thresholds")
    n_cells = qualifying["battery_id"].nunique()
    rows = []
    for threshold in SENSITIVITY_THRESHOLDS:
        crossing_index = find_crossings(qualifying, threshold)
        rows.append({"threshold": threshold, "crossing_cells": len(crossing_index), "of": n_cells})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))

    workable = table[table["crossing_cells"] >= MIN_WORKABLE_CROSSING_CELLS]
    default_row = table[table["threshold"] == DEFAULT_SOH_THRESHOLD].iloc[0]
    if default_row["crossing_cells"] < MIN_WORKABLE_CROSSING_CELLS:
        print(
            f"\n{DEFAULT_SOH_THRESHOLD} leaves only {default_row['crossing_cells']} of {n_cells} crossing "
            f"cells -- too few to train on. "
        )
        if not workable.empty:
            recommended = workable.iloc[workable["crossing_cells"].argmax()]["threshold"]
            print(f"{recommended} gives a workable population ({workable['crossing_cells'].max()} crossing cells).")
        else:
            print("None of the tested thresholds reach a workable population.")
    else:
        print(
            f"\n{DEFAULT_SOH_THRESHOLD} gives {int(default_row['crossing_cells'])} of {n_cells} crossing cells -- "
            f"workable, no threshold change forced by cell count alone."
        )
    return DEFAULT_SOH_THRESHOLD


def plot_crossing_budget(qualifying: pd.DataFrame, crossing_index: dict[str, int]) -> None:
    cell_order = qualifying.groupby("battery_id")["discharge_cycle_index"].max().sort_values().index.tolist()

    fig, ax = plt.subplots(figsize=(11, 8))
    for row_position, battery_id in enumerate(cell_order):
        n_cycles = int(qualifying.loc[qualifying["battery_id"] == battery_id, "discharge_cycle_index"].max()) + 1
        ax.barh(row_position, n_cycles, color="lightgrey", edgecolor="black", height=0.6)

        if battery_id in crossing_index:
            ci = crossing_index[battery_id]
            window_start = max(0, ci - PLOT_LEAD_TIME)
            ax.barh(row_position, ci - window_start, left=window_start, color="orange", height=0.6, label="warning window" if row_position == 0 else None)
            ax.plot([ci, ci], [row_position - 0.35, row_position + 0.35], color="red", linewidth=2, label="crossing" if row_position == 0 else None)

    ax.set_yticks(range(len(cell_order)))
    ax.set_yticklabels(cell_order, fontsize=8)
    ax.set_xlabel("discharge_cycle_index")
    ax.set_title(f"NASA: crossing budget per cell (soh < {DEFAULT_SOH_THRESHOLD}, {PLOT_LEAD_TIME}-cycle warning window shaded)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "s06a_crossing_budget.png")
    plt.close(fig)


def recommend(cycle_table: pd.DataFrame, crossing_index: dict[str, int], qualifying: pd.DataFrame) -> None:
    section("Recommendation")

    artifact_cells = []
    for battery_id, ci in crossing_index.items():
        if ci != 0:
            continue
        group = qualifying[qualifying["battery_id"] == battery_id].sort_values("discharge_cycle_index")
        after_first = group[group["discharge_cycle_index"] > 0]
        if not after_first.empty and after_first["soh"].iloc[0] >= DEFAULT_SOH_THRESHOLD:
            artifact_cells.append(battery_id)

    balanced = cycle_table.copy()
    balanced["imbalance"] = (balanced[["normal_pct", "warning_pct", "runaway_pct"]] - 100 / 3).abs().sum(axis=1)
    feasible = balanced[balanced["cells_with_insufficient_history"] == balanced["cells_with_insufficient_history"].min()]
    best = feasible.loc[feasible["imbalance"].idxmin()]

    print(
        f"Best cycle-level three-class balance among lead times with the fewest starved cells: "
        f"lead_time={int(best['lead_time'])} (normal={best['normal_pct']:.1f}%, warning={best['warning_pct']:.1f}%, "
        f"runaway={best['runaway_pct']:.1f}%, {int(best['cells_with_insufficient_history'])} cell(s) with "
        f"insufficient pre-crossing history)."
    )
    print(
        f"\nTradeoff: RUNAWAY sits around {cycle_table['runaway_pct'].iloc[0]:.0f}% at every lead time tested "
        f"because it only depends on crossing_index, not lead_time -- and {len(artifact_cells)} of "
        f"{len(crossing_index)} crossing cells ({artifact_cells}) contribute cycles that are 100% RUNAWAY "
        f"purely from a noisy first-cycle reading, not real degradation. That inflates the RUNAWAY share "
        f"regardless of lead time; excluding those cells (or fixing the crossing definition to require "
        f"the dip to persist) would meaningfully rebalance the classes and should happen before labels "
        f"are finalized -- longer lead times buy more WARNING cycles at the cost of more cells running out "
        f"of pre-crossing history to fill the window (see cells_with_insufficient_history above)."
    )


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    qualifying = load_scope()
    crossing_index = crossing_inventory(qualifying, DEFAULT_SOH_THRESHOLD)

    cycle_table = lead_time_feasibility(qualifying, crossing_index)
    measurement_level_feasibility(qualifying, crossing_index)

    threshold_sensitivity(qualifying)

    plot_crossing_budget(qualifying, crossing_index)
    print(f"\nsaved plot to {FIGURES_DIR / 's06a_crossing_budget.png'}")

    recommend(cycle_table, crossing_index, qualifying)


if __name__ == "__main__":
    main()
