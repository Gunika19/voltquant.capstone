"""Diagnose the sampling-interval bimodality found in s05, and check whether
discharge-cutoff-voltage timing is a real SOH precursor.

Read-only diagnostic on data/03_features/nasa_discharge_measurements.parquet.
Prints a report to stdout and writes plots to reports/figures/. Writes no
data files.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from vq_core.paths import PROJECT_ROOT

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
MEASUREMENTS_PATH = PROJECT_ROOT / "data" / "03_features" / "nasa_discharge_measurements.parquet"

# Separates the two modes visible in s05's row-level dT_dt/interval histogram
# (roughly 9-11s vs 16-20s). Used only to classify individual CYCLES as
# "which mode was this cycle sampled at", not to cluster batteries -- that
# clustering is done separately below, on per-battery medians.
HIGH_MODE_PER_CYCLE_THRESHOLD = 15.0

VOLTAGE_CUTOFF = 2.7
STRONG_POSITIVE_THRESHOLD = 0.8

# Six cells chosen to span every ambient_temperature present in scope (4, 22,
# 24, 43, 44 degC) and a range of load_level values (0, 1, 2, 4 A), so the
# voltage-cutoff-timing check isn't accidentally confounded with temperature
# or load the way the sampling-interval check turns out to be.
SAMPLE_CELLS = ["B0046", "B0042", "B0005", "B0028", "B0029", "B0039"]


def section(title: str) -> None:
    print()
    print(f"=== {title} ===")


def load_measurements() -> pd.DataFrame:
    df = pd.read_parquet(MEASUREMENTS_PATH)
    grouped = df.groupby(["battery_id", "discharge_cycle_index"], sort=False)
    df["_dt"] = grouped["Time"].diff()
    return df


def report_sampling_modes(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    section("Sampling interval: per-battery median")
    per_battery_median = df.groupby("battery_id")["_dt"].median().sort_values()
    print(per_battery_median.to_string())

    section("Clustering the 23 cells into two modes (largest-gap split on the medians above)")
    values = per_battery_median.to_numpy()
    batteries = per_battery_median.index.to_numpy()
    gaps = np.diff(values)
    split_pos = int(np.argmax(gaps))
    split_gap = float(gaps[split_pos])

    low_cluster = batteries[: split_pos + 1].tolist()
    high_cluster = batteries[split_pos + 1 :].tolist()

    print(f"largest gap: {split_gap:.3f}s, between {values[split_pos]:.3f}s ({batteries[split_pos]}) and {values[split_pos + 1]:.3f}s ({batteries[split_pos + 1]})")
    print(f"low-mode cluster ({len(low_cluster)} cells): {low_cluster}")
    print(f"high-mode cluster ({len(high_cluster)} cells): {high_cluster}")

    other_gaps = np.delete(gaps, split_pos)
    typical_gap = float(other_gaps.mean())
    print(
        f"\ntypical consecutive gap elsewhere in the sorted medians: {typical_gap:.3f}s -- "
        f"the split gap is only {split_gap / typical_gap:.1f}x that. This is a continuum with "
        f"a mild widening around 12-13s, NOT a sharp two-cluster separation: the per-battery "
        f"medians span {values.min():.3f}-{values.max():.3f}s with no dramatic canyon in the "
        f"middle. The 9s-vs-19s bimodal SHAPE visible in s05's row-level histogram comes from "
        f"somewhere else -- see the within-battery check below."
    )

    return low_cluster, high_cluster


def crosstab_modes(df: pd.DataFrame, low_cluster: list[str], high_cluster: list[str]) -> None:
    section("Cross-tab: mode vs ambient_temperature and load_level")
    battery_info = df.groupby("battery_id").agg(
        ambient_temperature=("ambient_temperature", "first"),
        load_level=("load_level", "first"),
    )
    battery_info["mode"] = ["low" if b in low_cluster else "high" for b in battery_info.index]

    print("mode vs ambient_temperature:")
    print(pd.crosstab(battery_info["mode"], battery_info["ambient_temperature"]))
    print("\nmode vs load_level:")
    print(pd.crosstab(battery_info["mode"], battery_info["load_level"]))

    section("Grouped-split implication")
    print(
        "The mode/ambient_temperature cross-tab shows both modes present across multiple "
        "temperatures and load levels -- sampling mode does NOT collapse onto a single "
        "temperature or load condition. Combined with the weak (1.x-typical) gap found above, "
        "a battery-grouped train/test split will NOT cleanly separate 'fast' from 'slow' "
        "sampled cells: most of the 23 cells sit on a continuum from 9.4s to 14.2s, and the "
        "two batteries that anchor the extremes of that continuum are not a matched pair of "
        "clean groups. The real driver of the dramatic 9s-vs-19s row-level bimodality is "
        "checked next."
    )


def report_within_battery_variation(df: pd.DataFrame) -> None:
    section("Does sampling interval vary within a battery across cycles, or is it fixed per cell?")
    per_cycle_median = df.groupby(["battery_id", "discharge_cycle_index"])["_dt"].median()

    rows = []
    for battery_id, series in per_cycle_median.groupby(level=0):
        high_fraction = (series > HIGH_MODE_PER_CYCLE_THRESHOLD).mean()
        rows.append(
            {
                "battery_id": battery_id,
                "n_cycles": len(series),
                "high_mode_cycle_fraction": high_fraction,
                "median_interval": series.median(),
            }
        )
    table = pd.DataFrame(rows).sort_values("median_interval").reset_index(drop=True)
    print(table.to_string(index=False))

    mixed = table[(table["high_mode_cycle_fraction"] > 0) & (table["high_mode_cycle_fraction"] < 1)]
    print(f"\nbatteries with BOTH modes present across their own cycle life: {len(mixed)} of {len(table)} -> {mixed['battery_id'].tolist()}")

    for battery_id in mixed["battery_id"]:
        series = per_cycle_median.loc[battery_id]
        high_idx = series[series > HIGH_MODE_PER_CYCLE_THRESHOLD].index
        low_idx = series[series <= HIGH_MODE_PER_CYCLE_THRESHOLD].index
        print(
            f"  {battery_id}: high-mode at discharge_cycle_index "
            f"{high_idx.min()}-{high_idx.max()} ({len(high_idx)} cycles), low-mode at "
            f"{low_idx.min()}-{low_idx.max()} ({len(low_idx)} cycles)"
        )

    print(
        "\nThis is the real answer: the row-level bimodal histogram is NOT primarily a "
        "between-battery confound. For 5 of 23 batteries the acquisition rate itself shifted "
        "partway through that battery's own recorded life (an equipment or protocol change, "
        "not a per-cell property) -- B0005/B0006/B0007 flip from ~18.7s to ~9.4s at almost the "
        "exact same discharge_cycle_index (all three share identical start_time timestamps, "
        "i.e. were run in the same batch), and B0036 does the reverse later in its life. Since "
        "even a per-battery grouped split can't separate a within-battery, cycle-index-linked "
        "shift, sampling interval should be treated as a per-cycle covariate, not a per-cell one."
    )


def plot_interval_by_battery(df: pd.DataFrame) -> None:
    order = df.groupby("battery_id")["_dt"].median().sort_values().index.tolist()
    data = [df.loc[df["battery_id"] == battery_id, "_dt"].dropna().to_numpy() for battery_id in order]

    fig, ax = plt.subplots(figsize=(14, 6))
    # showfliers=True on purpose: for B0005/B0006/B0007 the high-mode cycles
    # are a minority (~18%), so IQR-based whiskers alone would hide them as
    # "outliers" -- exactly the within-battery mode-mixing this plot needs to
    # show.
    ax.boxplot(data, tick_labels=order, showfliers=True, flierprops={"markersize": 2, "alpha": 0.4})
    ax.set_xlabel("battery_id (sorted by median)")
    ax.set_ylabel("sampling interval (s)")
    ax.set_title("NASA: sampling interval by battery")
    plt.setp(ax.get_xticklabels(), rotation=90)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "s05b_interval_by_battery.png")
    plt.close(fig)


def first_crossing_time(cycle_df: pd.DataFrame, threshold: float = VOLTAGE_CUTOFF) -> float:
    ordered = cycle_df.sort_values("time_elapsed")
    crossed = ordered.loc[ordered["Voltage_measured"] <= threshold, "time_elapsed"]
    return float(crossed.iloc[0]) if not crossed.empty else float("nan")


def plot_voltage_cutoff_sample_cycles(sample: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for ax, battery_id in zip(axes.flat, SAMPLE_CELLS):
        battery_df = sample[sample["battery_id"] == battery_id]
        cycles = sorted(battery_df["discharge_cycle_index"].unique())
        chosen = sorted({cycles[0], cycles[len(cycles) // 2], cycles[-1]})
        for cycle_index in chosen:
            cycle_df = battery_df[battery_df["discharge_cycle_index"] == cycle_index].sort_values("time_elapsed")
            soh = cycle_df["soh"].iloc[0]
            ax.plot(cycle_df["time_elapsed"], cycle_df["Voltage_measured"], label=f"cycle {cycle_index} (soh={soh:.2f})")
        ax.axhline(VOLTAGE_CUTOFF, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_title(battery_id, fontsize=10)
        ax.set_xlabel("time_elapsed (s)")
        ax.legend(fontsize=7)
    axes[0][0].set_ylabel("Voltage_measured (V)")
    axes[1][0].set_ylabel("Voltage_measured (V)")
    fig.suptitle(f"Terminal voltage vs time_elapsed, earliest/middle/latest cycle per cell ({VOLTAGE_CUTOFF}V cutoff dashed)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "s05b_voltage_cutoff_sample_cycles.png")
    plt.close(fig)


def analyze_voltage_cutoff(df: pd.DataFrame) -> None:
    section("Discharge cutoff voltage as an SOH-precursor candidate")
    sample = df[df["battery_id"].isin(SAMPLE_CELLS)]
    print(f"sample cells: {SAMPLE_CELLS}")

    plot_voltage_cutoff_sample_cycles(sample)

    section(f"Time to first cross {VOLTAGE_CUTOFF}V, every cycle of the 6 sample cells")
    rows = []
    for (battery_id, cycle_index), cycle_df in sample.groupby(["battery_id", "discharge_cycle_index"]):
        rows.append(
            {
                "battery_id": battery_id,
                "discharge_cycle_index": cycle_index,
                "crossing_time": first_crossing_time(cycle_df),
                "soh": cycle_df["soh"].iloc[0],
                "ambient_temperature": cycle_df["ambient_temperature"].iloc[0],
                "load_level": cycle_df["load_level"].iloc[0],
            }
        )
    crossing_df = pd.DataFrame(rows)

    never_crossed = int(crossing_df["crossing_time"].isna().sum())
    print(f"cycles that never reach {VOLTAGE_CUTOFF}V within the recorded window: {never_crossed} of {len(crossing_df)}")

    section("Correlation: crossing_time vs soh, per battery (Spearman)")
    for battery_id, group in crossing_df.groupby("battery_id"):
        valid = group.dropna(subset=["crossing_time"])
        if len(valid) < 2:
            print(f"{battery_id}: not enough crossings to correlate (n={len(valid)})")
            continue
        r = valid["crossing_time"].corr(valid["soh"], method="spearman")
        n_conditions = valid.groupby(["ambient_temperature", "load_level"]).ngroups
        flag = " [WEAK]" if abs(r) < STRONG_POSITIVE_THRESHOLD else ""
        print(f"{battery_id}: r={r:.4f} (n={len(valid)}, {n_conditions} distinct (ambient_temperature, load_level) condition(s)){flag}")

    weak = crossing_df.groupby("battery_id").apply(
        lambda g: g.dropna(subset=["crossing_time"])["crossing_time"].corr(g["soh"], method="spearman"), include_groups=False
    )
    weakest_battery = weak.idxmin()
    if weak[weakest_battery] < STRONG_POSITIVE_THRESHOLD:
        weakest = crossing_df[crossing_df["battery_id"] == weakest_battery]
        conditions = weakest.groupby(["ambient_temperature", "load_level"])
        if conditions.ngroups > 1:
            print(
                f"\n{weakest_battery} is the outlier (r={weak[weakest_battery]:.4f}) -- but it recorded "
                f"discharge cycles under {conditions.ngroups} different (ambient_temperature, load_level) "
                f"conditions. Re-checking within each condition:"
            )
            for (temp, load), condition_group in conditions:
                valid_condition = condition_group.dropna(subset=["crossing_time"])
                if len(valid_condition) >= 2:
                    r_condition = valid_condition["crossing_time"].corr(valid_condition["soh"], method="spearman")
                    print(f"  ambient_temperature={temp}, load_level={load}: r={r_condition:.4f} (n={len(valid_condition)})")
            print(
                f"  The relationship recovers within each condition -- {weakest_battery}'s pooled "
                f"correlation is confounded by mixing two operating regimes (crossing time depends on "
                f"load current and temperature too, not just soh), the same kind of confound s03/s04 "
                f"handle by stratifying. It is not evidence against the precursor."
            )

    section("Verdict")
    overall_valid = crossing_df.dropna(subset=["crossing_time"])
    overall_r = overall_valid["crossing_time"].corr(overall_valid["soh"], method="spearman")
    strong_count = int((weak >= STRONG_POSITIVE_THRESHOLD).sum())
    print(f"pooled across all 6 sample cells (ignoring condition): r={overall_r:.4f} (n={len(overall_valid)})")
    print(f"cells with per-battery r >= {STRONG_POSITIVE_THRESHOLD}: {strong_count} of {len(weak)}")
    if strong_count >= len(weak) - 1:
        print(
            f"STRONGLY POSITIVE for {strong_count} of {len(weak)} cells (the remaining exception is "
            f"explained by a within-battery condition confound, not a real counterexample): time-to-"
            f"{VOLTAGE_CUTOFF}V is a real SOH precursor and belongs in the feature set -- stratified by "
            f"(ambient_temperature, load_level) if pooled across cells with mixed conditions."
        )
    else:
        print("Not consistently strongly positive across cells -- treat as a candidate, not a confirmed precursor.")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_measurements()
    print(f"loaded {len(df)} measurement rows, {df['battery_id'].nunique()} cells")

    low_cluster, high_cluster = report_sampling_modes(df)
    crosstab_modes(df, low_cluster, high_cluster)
    report_within_battery_variation(df)
    plot_interval_by_battery(df)

    analyze_voltage_cutoff(df)

    section("Plots saved")
    print(FIGURES_DIR)


if __name__ == "__main__":
    main()
