"""Build the three-class early-warning label and join it onto measurements.

Reads data/02_intermediate/nasa_discharge_cycles.parquet and
data/03_features/nasa_discharge_measurements.parquet, writes
data/03_features/nasa_labelled_measurements.parquet.

Label scheme (see vq_core.schema for the constants and their justification):
  2 (runaway) : discharge_cycle_index >= sustained_crossing_index
  1 (warning) : sustained_crossing_index - WARNING_LEAD_CYCLES <= discharge_cycle_index
                < sustained_crossing_index
  0 (normal)  : everything else, including every cycle of a cell that never
                has a sustained crossing

"Sustained crossing" (as opposed to the naive first-crossing s06a used) is
the first discharge_cycle_index where soh < SOH_EOL_THRESHOLD and it stays
below threshold for the next CROSSING_PERSISTENCE_CYCLES consecutive valid
cycles. This is deliberately generic (no hardcoded battery names) -- see
`sustained_crossing_index()`.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vq_core.guards import assert_no_label_leakage
from vq_core.paths import PROJECT_ROOT
from vq_core.schema import CROSSING_PERSISTENCE_CYCLES, LABELS, SOH_EOL_THRESHOLD, WARNING_LEAD_CYCLES

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
CYCLES_PATH = PROJECT_ROOT / "data" / "02_intermediate" / "nasa_discharge_cycles.parquet"
MEASUREMENTS_PATH = PROJECT_ROOT / "data" / "03_features" / "nasa_discharge_measurements.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "03_features" / "nasa_labelled_measurements.parquet"

SCOPE_VERDICTS = ["CLEAN", "STRATIFIED"]

# Columns that must never survive into a feature-only view, on top of
# whatever vq_core.schema.LABEL_ONLY_FEATURES already knows about -- these
# are the label itself and its direct derivatives, computed only in this
# script.
LABEL_DERIVED_COLUMNS = ["label", "cycles_to_crossing"]


def section(title: str) -> None:
    print()
    print(f"=== {title} ===")


def load_scope() -> pd.DataFrame:
    section("Scope")
    cycles = pd.read_parquet(CYCLES_PATH)
    qualifying = cycles[cycles["verdict"].isin(SCOPE_VERDICTS) & (~cycles["invalid_measurement"])].copy()

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


def naive_crossing_index(qualifying: pd.DataFrame, threshold: float) -> dict[str, int]:
    crossing_index = {}
    for battery_id, group in qualifying.groupby("battery_id"):
        group = group.sort_values("discharge_cycle_index")
        crossed = group.loc[group["soh"] < threshold, "discharge_cycle_index"]
        if not crossed.empty:
            crossing_index[battery_id] = int(crossed.iloc[0])
    return crossing_index


def sustained_crossing_index(qualifying: pd.DataFrame, threshold: float, persistence: int) -> dict[str, int]:
    """First discharge_cycle_index where soh < threshold for `persistence` consecutive valid cycles after it.

    "Consecutive valid cycles" means positional neighbours in this cell's own
    sorted, already-scoped cycle sequence -- not discharge_cycle_index
    arithmetic, since invalid/out-of-scope cycles have already been removed
    and would otherwise create phantom gaps.

    Near the end of a cell's recorded life, fewer than `persistence` future
    cycles may exist. The window is truncated rather than discarded in that
    case: if every remaining recorded cycle from the candidate onward stays
    below threshold, there is no observed rebound to disprove sustained
    decline, so it counts as sustained. Requiring a full unbroken window here
    would erase genuine end-of-life crossings just because recording stopped
    a few cycles early, which is worse than the truncation's own edge risk.
    """
    crossing_index = {}
    for battery_id, group in qualifying.groupby("battery_id"):
        group = group.sort_values("discharge_cycle_index")
        below = group["soh"].to_numpy() < threshold
        indices = group["discharge_cycle_index"].to_numpy()
        for position in range(len(below)):
            window = below[position : position + persistence + 1]
            if window.all():
                crossing_index[battery_id] = int(indices[position])
                break
    return crossing_index


def report_naive_vs_sustained(naive: dict[str, int], sustained: dict[str, int], all_cells: list[str]) -> dict[str, int]:
    section(f"Naive vs sustained crossing (soh < {SOH_EOL_THRESHOLD}, persistence = {CROSSING_PERSISTENCE_CYCLES} cycles)")
    rows = []
    for battery_id in all_cells:
        n = naive.get(battery_id)
        s = sustained.get(battery_id)
        rows.append({"battery_id": battery_id, "naive_crossing": n, "sustained_crossing": s, "moved": n != s})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))

    moved = table[table["moved"]]
    lost_crossing = sorted(moved.loc[moved["sustained_crossing"].isna(), "battery_id"])
    shifted = moved[moved["sustained_crossing"].notna()]
    print(f"\n{len(moved)} of {len(table)} cells moved.")
    if lost_crossing:
        print(
            f"{len(lost_crossing)} cell(s) lost their crossing entirely -- their naive crossing never persisted "
            f"3 consecutive cycles, so they are now non-crossing cells: {lost_crossing}"
        )
    if not shifted.empty:
        for _, row in shifted.sort_values("battery_id").iterrows():
            print(f"  {row['battery_id']}: naive={row['naive_crossing']} -> sustained={int(row['sustained_crossing'])}")

    return sustained


def assign_labels(qualifying: pd.DataFrame, crossing_index: dict[str, int]) -> pd.DataFrame:
    section("Label assignment")
    labelled = qualifying[["battery_id", "discharge_cycle_index"]].copy()
    labelled["_crossing_index"] = labelled["battery_id"].map(crossing_index)

    has_crossing = labelled["_crossing_index"].notna()
    is_runaway = has_crossing & (labelled["discharge_cycle_index"] >= labelled["_crossing_index"])
    is_warning = (
        has_crossing
        & ~is_runaway
        & (labelled["discharge_cycle_index"] >= labelled["_crossing_index"] - WARNING_LEAD_CYCLES)
    )

    labelled["label"] = 0
    labelled.loc[is_warning, "label"] = 1
    labelled.loc[is_runaway, "label"] = 2
    labelled["cycles_to_crossing"] = np.where(
        has_crossing, labelled["discharge_cycle_index"] - labelled["_crossing_index"], np.nan
    )
    labelled = labelled.drop(columns="_crossing_index")

    print("cycle-level label counts:")
    counts = labelled["label"].value_counts().sort_index()
    for label, count in counts.items():
        print(f"  {label} ({LABELS[label]}): {count} ({100 * count / len(labelled):.1f}%)")

    return labelled


def report_per_cell_and_no_warning_cells(labelled: pd.DataFrame, crossing_index: dict[str, int]) -> None:
    section("Per-cell label counts")
    per_cell = (
        labelled.groupby("battery_id")["label"]
        .value_counts()
        .unstack(fill_value=0)
        .rename(columns=LABELS)
        .reindex(columns=[LABELS[0], LABELS[1], LABELS[2]], fill_value=0)
        .sort_index()
    )
    print(per_cell.to_string())

    no_warning = sorted(
        battery_id
        for battery_id in crossing_index
        if per_cell.loc[battery_id, LABELS[1]] == 0
    )
    if no_warning:
        print(
            f"\n{len(no_warning)} crossing cell(s) contribute zero warning-labelled cycles -- their sustained "
            f"crossing is within {WARNING_LEAD_CYCLES} cycles of the start of their recorded life, so the "
            f"warning window runs off the start: {no_warning}"
        )
    else:
        print("\nevery crossing cell contributes at least one warning-labelled cycle.")


def build_labelled_measurements(labelled: pd.DataFrame) -> pd.DataFrame:
    section("Join to measurements")
    measurements = pd.read_parquet(MEASUREMENTS_PATH)
    before = len(measurements)

    label_frame = labelled.drop_duplicates(subset=["battery_id", "discharge_cycle_index"])
    joined = measurements.merge(
        label_frame,
        on=["battery_id", "discharge_cycle_index"],
        how="inner",
        validate="many_to_one",
    )

    dropped = before - len(joined)
    print(
        f"measurement rows before join: {before}, after: {len(joined)} ({dropped} dropped -- rows for cycles "
        f"outside label scope, e.g. B0040's non-dominant-stratum cycles that s05 did not filter out)"
    )
    return joined


def report_class_balance(labelled: pd.DataFrame, joined: pd.DataFrame) -> None:
    section("Three-class balance")
    print("cycle level:")
    cycle_counts = labelled["label"].value_counts().sort_index()
    for label, count in cycle_counts.items():
        print(f"  {label} ({LABELS[label]}): {count} ({100 * count / len(labelled):.1f}%)")

    print("\nmeasurement level:")
    meas_counts = joined["label"].value_counts().sort_index()
    for label, count in meas_counts.items():
        print(f"  {label} ({LABELS[label]}): {count} ({100 * count / len(joined):.1f}%)")


def check_no_leakage(joined: pd.DataFrame) -> list[str]:
    section("Leakage guard on feature-only view")
    drop_columns = [
        column
        for column in joined.columns
        if column in ("label", "soh", "soh_pct", "cycles_to_crossing")
        or column.lower().startswith("capacity")
        or column.lower().startswith("time_to_")
    ]
    feature_view = joined.drop(columns=drop_columns)
    print(f"dropped for feature-only view: {drop_columns}")
    print(f"feature-only view columns: {list(feature_view.columns)}")

    assert_no_label_leakage(feature_view)
    print("assert_no_label_leakage: PASSED -- feature-only view carries no label-only column.")
    return drop_columns


def plot_label_timeline(labelled: pd.DataFrame, crossing_index: dict[str, int]) -> None:
    cell_order = labelled.groupby("battery_id")["discharge_cycle_index"].max().sort_values().index.tolist()
    colors = {0: "lightgrey", 1: "orange", 2: "red"}

    fig, ax = plt.subplots(figsize=(11, 8))
    for row_position, battery_id in enumerate(cell_order):
        group = labelled[labelled["battery_id"] == battery_id].sort_values("discharge_cycle_index")
        for label, color in colors.items():
            segment = group[group["label"] == label]
            if segment.empty:
                continue
            ax.scatter(
                segment["discharge_cycle_index"],
                [row_position] * len(segment),
                color=color,
                marker="s",
                s=8,
                label=LABELS[label] if row_position == 0 else None,
            )
        if battery_id in crossing_index:
            ci = crossing_index[battery_id]
            ax.plot([ci, ci], [row_position - 0.35, row_position + 0.35], color="black", linewidth=1)

    ax.set_yticks(range(len(cell_order)))
    ax.set_yticklabels(cell_order, fontsize=8)
    ax.set_xlabel("discharge_cycle_index")
    ax.set_title(f"NASA: label timeline per cell (soh < {SOH_EOL_THRESHOLD}, sustained {CROSSING_PERSISTENCE_CYCLES} cycles)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "s06_label_timeline.png")
    plt.close(fig)
    print(f"saved plot to {FIGURES_DIR / 's06_label_timeline.png'}")


def plot_class_balance(labelled: pd.DataFrame, joined: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, frame, title in [(axes[0], labelled, "cycle level"), (axes[1], joined, "measurement level")]:
        counts = frame["label"].value_counts().sort_index()
        counts = counts.reindex([0, 1, 2], fill_value=0)
        ax.bar([LABELS[i] for i in counts.index], counts.values, color=["lightgrey", "orange", "red"])
        ax.set_title(title)
        ax.set_ylabel("count")
        for i, value in enumerate(counts.values):
            ax.text(i, value, f"{100 * value / counts.sum():.1f}%", ha="center", va="bottom")

    fig.suptitle("NASA: three-class balance")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "s06_class_balance.png")
    plt.close(fig)
    print(f"saved plot to {FIGURES_DIR / 's06_class_balance.png'}")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    qualifying = load_scope()

    naive = naive_crossing_index(qualifying, SOH_EOL_THRESHOLD)
    sustained = sustained_crossing_index(qualifying, SOH_EOL_THRESHOLD, CROSSING_PERSISTENCE_CYCLES)
    all_cells = sorted(qualifying["battery_id"].unique())
    crossing_index = report_naive_vs_sustained(naive, sustained, all_cells)

    labelled = assign_labels(qualifying, crossing_index)
    report_per_cell_and_no_warning_cells(labelled, crossing_index)

    joined = build_labelled_measurements(labelled)
    report_class_balance(labelled, joined)
    check_no_leakage(joined)

    plot_label_timeline(labelled, crossing_index)
    plot_class_balance(labelled, joined)

    joined.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nwrote {OUTPUT_PATH} ({joined.shape[0]} rows, {joined.shape[1]} columns)")


if __name__ == "__main__":
    main()
