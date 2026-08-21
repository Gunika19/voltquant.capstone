"""Diagnose why some NASA batteries failed the s02 monotonicity gate.

Tests two confounds -- ambient temperature and discharge load current -- by
recomputing the cycle_index/Capacity Spearman correlation within strata, then
visually inspects whatever is left unexplained. Read-only: prints a report to
stdout and writes plots to reports/figures/. Writes no data files.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from vq_core.paths import PROJECT_ROOT, RAW_NASA
from vq_data.s02_verify_nasa import (
    METADATA_SEARCH_DEPTH,
    find_metadata_csv,
    find_per_cycle_dir,
    parse_start_time,
)

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

STRONG_NEGATIVE_THRESHOLD = -0.8

# s02 asked for >=15 cycles for the (battery, ambient_temperature) strata; we
# hold hypothesis 2's finer (battery, ambient_temperature, load_level) strata
# to the same bar so "strongly negative" means the same thing in both tables.
MIN_STRATUM_CYCLES = 15

MAX_SUBPLOTS_PER_FIGURE = 12
SUBPLOT_GRID_COLUMNS = 3


def section(title: str) -> None:
    print()
    print(f"=== {title} ===")


def load_metadata_with_cycle_index() -> tuple[pd.DataFrame, Path]:
    metadata_path = find_metadata_csv(RAW_NASA, METADATA_SEARCH_DEPTH)
    if metadata_path is None:
        raise FileNotFoundError(f"metadata.csv not found within {METADATA_SEARCH_DEPTH} levels of {RAW_NASA}")

    df = pd.read_csv(metadata_path)
    df["_start_dt"] = df["start_time"].map(parse_start_time)
    df["cycle_index"] = df.groupby("battery_id")["_start_dt"].rank(method="first").astype(int) - 1
    return df, metadata_path.parent


def prepare_discharge(df: pd.DataFrame) -> pd.DataFrame:
    section("Discharge rows: dropping missing Capacity")
    df = df.copy()
    df["Capacity_num"] = pd.to_numeric(df["Capacity"], errors="coerce")
    discharge = df[df["type"] == "discharge"].copy()

    missing = discharge[discharge["Capacity_num"].isna()]
    if missing.empty:
        print("no rows dropped -- Capacity present on every discharge row")
    else:
        print("dropped rows (missing Capacity) per battery:")
        print(missing.groupby("battery_id").size())
    print(f"\ntotal dropped: {len(missing)} of {len(discharge)} discharge rows")

    return discharge.dropna(subset=["Capacity_num"]).copy()


def pooled_spearman(discharge: pd.DataFrame) -> pd.Series:
    correlations = {}
    for battery_id, group in discharge.groupby("battery_id"):
        if group["cycle_index"].nunique() > 1:
            correlations[battery_id] = group["cycle_index"].corr(group["Capacity_num"], method="spearman")
    return pd.Series(correlations, name="pooled_spearman").sort_values()


def stratified_spearman(
    discharge: pd.DataFrame,
    group_keys: list[str],
    eligible_batteries: set | None = None,
    min_cycles: int = MIN_STRATUM_CYCLES,
):
    """Spearman per stratum of `group_keys`, kept only if it has >= min_cycles rows."""
    per_battery: dict[str, list[float]] = {}
    rows = []
    for key, group in discharge.groupby(group_keys):
        battery_id = key[0] if isinstance(key, tuple) else key
        if eligible_batteries is not None and battery_id not in eligible_batteries:
            continue
        if len(group) < min_cycles or group["cycle_index"].nunique() < 2:
            continue
        r = group["cycle_index"].corr(group["Capacity_num"], method="spearman")
        per_battery.setdefault(battery_id, []).append(r)
        row = dict(zip(group_keys, key if isinstance(key, tuple) else (key,)))
        row["n_cycles"] = len(group)
        row["spearman_r"] = r
        rows.append(row)
    return per_battery, pd.DataFrame(rows)


def report_confound_result(hypothesis_name: str, group_keys: list[str], strata_df: pd.DataFrame,
                            per_battery: dict, pooled: pd.Series, failing: list[str]) -> None:
    if strata_df.empty:
        print(f"\nno {group_keys} strata reached the {MIN_STRATUM_CYCLES}-cycle minimum")
    else:
        print(f"\n{tuple(group_keys)} strata with >= {MIN_STRATUM_CYCLES} cycles:")
        print(strata_df.sort_values(group_keys).to_string(index=False))

    section(f"{hypothesis_name}: pooled vs per-stratum comparison")
    for battery_id in pooled.index:
        per_stratum = per_battery.get(battery_id, [])
        formatted = ", ".join(f"{r:.3f}" for r in per_stratum) if per_stratum else "n/a"
        print(f"{battery_id}: pooled={pooled[battery_id]:.3f}  strata=[{formatted}]")

    became_clean = [
        battery_id
        for battery_id in failing
        if per_battery.get(battery_id) and min(per_battery[battery_id]) < STRONG_NEGATIVE_THRESHOLD
    ]
    print(
        f"\nof the {len(failing)} previously-failing batteries, {len(became_clean)} become "
        f"strongly negative (< {STRONG_NEGATIVE_THRESHOLD}) once stratified by {group_keys}: {became_clean}"
    )


def hypothesis_1_temperature(discharge: pd.DataFrame, pooled: pd.Series, failing: list[str]):
    section("Hypothesis 1: ambient temperature confound")
    print("distinct ambient_temperature values per battery (discharge rows):")
    print(discharge.groupby("battery_id")["ambient_temperature"].unique())

    per_battery, strata_df = stratified_spearman(discharge, ["battery_id", "ambient_temperature"])
    report_confound_result("Hypothesis 1", ["battery_id", "ambient_temperature"], strata_df, per_battery, pooled, failing)
    return per_battery


def extract_load_current(discharge: pd.DataFrame, per_cycle_dir: Path) -> pd.DataFrame:
    section("Hypothesis 2: extracting discharge load current")
    filenames = discharge["filename"].tolist()
    start = time.time()
    loads = []
    for filename in filenames:
        file_df = pd.read_csv(
            per_cycle_dir / filename, usecols=lambda c: c in ("Current_measured", "Current_load")
        )
        column = "Current_load" if "Current_load" in file_df.columns else "Current_measured"
        loads.append(file_df[column].abs().median())
    elapsed = time.time() - start

    discharge = discharge.copy()
    discharge["load_current"] = loads
    print(
        f"read {len(filenames)} discharge per-cycle files in {elapsed:.1f}s "
        f"(cached in memory as load_current, no data files written)"
    )
    return discharge


def hypothesis_2_load(discharge: pd.DataFrame, pooled: pd.Series, failing: list[str]):
    section("Hypothesis 2: discharge load confound")
    discharge["load_level"] = discharge["load_current"].round(1)

    load_variation = discharge.groupby("battery_id")["load_level"].nunique()
    varying_batteries = set(load_variation[load_variation > 1].index)
    print(
        f"batteries with >1 distinct load level: {len(varying_batteries)} "
        f"of {discharge['battery_id'].nunique()}"
    )
    print("\ndistinct load levels per battery:")
    print(discharge.groupby("battery_id")["load_level"].unique())

    if not varying_batteries:
        print("\nload current is constant per battery -- hypothesis 2 does not apply")
        return {}

    per_battery, strata_df = stratified_spearman(
        discharge, ["battery_id", "ambient_temperature", "load_level"], eligible_batteries=varying_batteries
    )
    report_confound_result(
        "Hypothesis 2", ["battery_id", "ambient_temperature", "load_level"], strata_df, per_battery, pooled, failing
    )
    return per_battery


def classify_verdicts(pooled: pd.Series, h1_strata: dict, h2_strata: dict):
    section("Per-battery verdict")
    verdicts = {}
    for battery_id, pooled_r in pooled.items():
        candidates = [pooled_r, *h1_strata.get(battery_id, []), *h2_strata.get(battery_id, [])]
        best_r = min(candidates)
        if pooled_r < STRONG_NEGATIVE_THRESHOLD:
            verdicts[battery_id] = "CLEAN"
        elif best_r < STRONG_NEGATIVE_THRESHOLD:
            verdicts[battery_id] = "STRATIFIED"
        else:
            verdicts[battery_id] = "ANOMALOUS"

    verdict_series = pd.Series(verdicts, name="verdict").sort_values()
    print(verdict_series.to_string())

    print("\ncounts:")
    print(verdict_series.value_counts())

    anomalous = sorted(b for b, v in verdicts.items() if v == "ANOMALOUS")
    print(f"\nANOMALOUS batteries ({len(anomalous)}): {anomalous}")
    return verdict_series, anomalous


def plot_anomalous_batteries(discharge: pd.DataFrame, anomalous: list[str]) -> None:
    section("Hypothesis 3: visual inspection of unexplained batteries")
    if not anomalous:
        print("no ANOMALOUS batteries -- nothing to plot")
        return

    temps = sorted(discharge["ambient_temperature"].unique())
    cmap = plt.get_cmap("viridis", len(temps))
    temp_color = {temp: cmap(i) for i, temp in enumerate(temps)}
    max_load = discharge["load_current"].max()

    pages = [
        anomalous[i : i + MAX_SUBPLOTS_PER_FIGURE] for i in range(0, len(anomalous), MAX_SUBPLOTS_PER_FIGURE)
    ]
    for page_number, battery_ids in enumerate(pages, start=1):
        n_cols = SUBPLOT_GRID_COLUMNS
        n_rows = -(-len(battery_ids) // n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.4 * n_rows), squeeze=False)

        for i, battery_id in enumerate(battery_ids):
            ax = axes[i // n_cols][i % n_cols]
            group = discharge[discharge["battery_id"] == battery_id].sort_values("cycle_index")
            colors = [temp_color[t] for t in group["ambient_temperature"]]
            sizes = 12 + 40 * (group["load_current"] / max_load)
            ax.scatter(group["cycle_index"], group["Capacity_num"], c=colors, s=sizes, alpha=0.8, edgecolors="none")
            ax.set_title(battery_id, fontsize=10)
            ax.set_xlabel("cycle_index", fontsize=8)
            ax.set_ylabel("Capacity", fontsize=8)
            ax.tick_params(labelsize=7)

        for j in range(len(battery_ids), n_rows * n_cols):
            axes[j // n_cols][j % n_cols].axis("off")

        fig.suptitle("ANOMALOUS batteries -- colour: ambient_temperature, size: load current")
        fig.tight_layout()
        out_path = FIGURES_DIR / f"s03_anomalous_batteries_page{page_number}.png"
        fig.savefig(out_path)
        plt.close(fig)
        print(f"saved {out_path}")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df, metadata_dir = load_metadata_with_cycle_index()
    sample_filename = next(iter(df["filename"].dropna()))
    per_cycle_dir = find_per_cycle_dir(metadata_dir, sample_filename)
    if per_cycle_dir is None:
        raise FileNotFoundError(f"could not locate the per-cycle data directory (looked for {sample_filename})")

    discharge = prepare_discharge(df)
    pooled = pooled_spearman(discharge)

    failing = pooled[pooled >= STRONG_NEGATIVE_THRESHOLD].index.tolist()
    section("Baseline: pooled Spearman (cycle_index vs Capacity), as in s02")
    print(pooled.to_string())
    print(
        f"\n{len(failing)} of {len(pooled)} batteries have pooled Spearman "
        f">= {STRONG_NEGATIVE_THRESHOLD}: {failing}"
    )

    h1_strata = hypothesis_1_temperature(discharge, pooled, failing)

    discharge = extract_load_current(discharge, per_cycle_dir)
    h2_strata = hypothesis_2_load(discharge, pooled, failing)

    verdicts, anomalous = classify_verdicts(pooled, h1_strata, h2_strata)

    plot_anomalous_batteries(discharge, anomalous)

    section("RESULT")
    counts = verdicts.value_counts()
    print(
        f"CLEAN: {counts.get('CLEAN', 0)}  "
        f"STRATIFIED: {counts.get('STRATIFIED', 0)}  "
        f"ANOMALOUS: {counts.get('ANOMALOUS', 0)}"
    )
    print(f"ANOMALOUS batteries: {anomalous}")


if __name__ == "__main__":
    main()
