"""Build the per-discharge-cycle NASA table with SOH ground truth.

Writes data/02_intermediate/nasa_discharge_cycles.parquet.

This table intentionally contains `capacity`, `capacity_ref`, `soh`, and
`soh_pct` -- it is the label-construction table, not a model-input table.
Running vq_core.guards.assert_no_label_leakage against it is EXPECTED to
raise LabelLeakageError; that's not a bug to work around here. Do not feed
this table directly to a model: derive features and labels from it
downstream, then call assert_no_label_leakage / assert_model_input_ready on
*that* table instead.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vq_core.paths import PROJECT_ROOT, RAW_NASA
from vq_data.s02_verify_nasa import METADATA_SEARCH_DEPTH, find_metadata_csv, find_per_cycle_dir, parse_start_time
from vq_data.s03_diagnose_capacity_trend import (
    STRONG_NEGATIVE_THRESHOLD,
    extract_load_current,
    pooled_spearman,
    stratified_spearman,
)

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
OUTPUT_PATH = PROJECT_ROOT / "data" / "02_intermediate" / "nasa_discharge_cycles.parquet"

CAPACITY_REF_TOP_N = 5
LOW_RATIO = 0.30
HIGH_RATIO = 1.10
STEP2_MIN_STRATUM_CYCLES = 8
MIN_VALID_CYCLES = 15

VERDICT_COLORS = {
    "CLEAN": "tab:green",
    "STRATIFIED": "tab:orange",
    "ANOMALOUS": "tab:red",
    "INSUFFICIENT": "tab:gray",
}


def section(title: str) -> None:
    print()
    print(f"=== {title} ===")


def load_discharge_rows() -> tuple[pd.DataFrame, Path]:
    metadata_path = find_metadata_csv(RAW_NASA, METADATA_SEARCH_DEPTH)
    if metadata_path is None:
        raise FileNotFoundError(f"metadata.csv not found within {METADATA_SEARCH_DEPTH} levels of {RAW_NASA}")

    df = pd.read_csv(metadata_path)
    df["_start_dt"] = df["start_time"].map(parse_start_time)
    df["cycle_index"] = df.groupby("battery_id")["_start_dt"].rank(method="first").astype(int) - 1
    df["Capacity_num"] = pd.to_numeric(df["Capacity"], errors="coerce")

    discharge = df[df["type"] == "discharge"].copy()

    sample_filename = next(iter(discharge["filename"].dropna()))
    per_cycle_dir = find_per_cycle_dir(metadata_path.parent, sample_filename)
    if per_cycle_dir is None:
        raise FileNotFoundError(f"could not locate the per-cycle data directory (looked for {sample_filename})")

    discharge = extract_load_current(discharge, per_cycle_dir)
    discharge["load_level"] = discharge["load_current"].round(1)
    return discharge, per_cycle_dir


def top5_median_capacity(rows: pd.DataFrame) -> float:
    positive = rows["Capacity_num"].dropna()
    positive = positive[positive > 0]
    if positive.empty:
        return float("nan")
    return positive.nlargest(CAPACITY_REF_TOP_N).median()


def capacity_ref_by_battery(discharge: pd.DataFrame) -> pd.Series:
    positive = discharge[discharge["Capacity_num"].notna() & (discharge["Capacity_num"] > 0)]
    ref = positive.groupby("battery_id")["Capacity_num"].apply(lambda s: s.nlargest(CAPACITY_REF_TOP_N).median())
    ref.name = "capacity_ref_step1"
    return ref


def step1_validity_filter(discharge: pd.DataFrame) -> pd.DataFrame:
    section("Step 1: validity filter")
    discharge = discharge.copy()

    capacity_ref = capacity_ref_by_battery(discharge)
    print(f"capacity_ref per battery (median of top {CAPACITY_REF_TOP_N} valid Capacity values):")
    print(capacity_ref.sort_values().to_string())

    discharge = discharge.merge(capacity_ref, how="left", left_on="battery_id", right_index=True)

    conditions = [
        discharge["Capacity_num"].isna(),
        discharge["Capacity_num"] <= 0,
        discharge["Capacity_num"] < LOW_RATIO * discharge["capacity_ref_step1"],
        discharge["Capacity_num"] > HIGH_RATIO * discharge["capacity_ref_step1"],
    ]
    choices = ["null", "non_positive", f"below_{int(LOW_RATIO * 100)}pct_of_ref", f"above_{int(HIGH_RATIO * 100)}pct_of_ref"]
    discharge["invalid_reason"] = np.select(conditions, choices, default="valid")
    discharge["invalid_measurement"] = discharge["invalid_reason"] != "valid"

    reason_table = pd.crosstab(discharge["battery_id"], discharge["invalid_reason"])
    print("\nflagged rows per battery by reason (columns exclude 'valid' if none occurred):")
    print(reason_table)

    total_flagged = int(discharge["invalid_measurement"].sum())
    print(f"\ntotal flagged invalid_measurement: {total_flagged} of {len(discharge)}")
    print("flag kept as a column -- rows are not dropped from the table.")

    return discharge


def classify(pooled: pd.Series, *strata_dicts: dict) -> dict:
    verdicts = {}
    for battery_id, pooled_r in pooled.items():
        candidates = [pooled_r]
        for strata in strata_dicts:
            candidates.extend(strata.get(battery_id, []))
        best_r = min(candidates)
        if pooled_r < STRONG_NEGATIVE_THRESHOLD:
            verdicts[battery_id] = "CLEAN"
        elif best_r < STRONG_NEGATIVE_THRESHOLD:
            verdicts[battery_id] = "STRATIFIED"
        else:
            verdicts[battery_id] = "ANOMALOUS"
    return verdicts


def s03_equivalent_verdicts(discharge: pd.DataFrame) -> dict:
    """Reproduce s03's classification (drop-null only, min 15 cycles, temp then temp+load)."""
    s03_discharge = discharge.dropna(subset=["Capacity_num"]).copy()
    pooled = pooled_spearman(s03_discharge)

    h1_strata, _ = stratified_spearman(s03_discharge, ["battery_id", "ambient_temperature"])

    load_variation = s03_discharge.groupby("battery_id")["load_level"].nunique()
    varying_batteries = set(load_variation[load_variation > 1].index)
    h2_strata, _ = stratified_spearman(
        s03_discharge, ["battery_id", "ambient_temperature", "load_level"], eligible_batteries=varying_batteries
    )

    return classify(pooled, h1_strata, h2_strata)


def step2_rediagnose(discharge: pd.DataFrame, verdicts_before: dict) -> tuple[pd.DataFrame, dict]:
    section("Step 2: re-diagnose (valid rows only)")

    valid = discharge[~discharge["invalid_measurement"]].copy()
    valid_counts = valid.groupby("battery_id").size()

    all_battery_ids = discharge["battery_id"].unique()
    insufficient = sorted(
        set(valid_counts[valid_counts < MIN_VALID_CYCLES].index) | set(b for b in all_battery_ids if b not in valid_counts.index)
    )
    print(f"batteries excluded as INSUFFICIENT (< {MIN_VALID_CYCLES} valid discharge cycles): {insufficient}")

    analyzable = valid[~valid["battery_id"].isin(insufficient)]
    pooled_after = pooled_spearman(analyzable)
    strata_after, strata_df = stratified_spearman(
        analyzable, ["battery_id", "ambient_temperature", "load_level"], min_cycles=STEP2_MIN_STRATUM_CYCLES
    )

    print(f"\n(battery_id, ambient_temperature, load_level) strata with >= {STEP2_MIN_STRATUM_CYCLES} valid cycles:")
    print(strata_df.sort_values(["battery_id", "ambient_temperature", "load_level"]).to_string(index=False) if not strata_df.empty else "  none")

    verdicts_after = classify(pooled_after, strata_after)
    for battery_id in insufficient:
        verdicts_after[battery_id] = "INSUFFICIENT"

    section("Step 2: before (s03) vs after (valid-only, min 8) verdict comparison")
    comparison = pd.DataFrame(
        {
            "s03_verdict": pd.Series(verdicts_before),
            "s04_verdict": pd.Series(verdicts_after),
        }
    ).sort_index()
    print(comparison.to_string())

    print("\ns03 verdict counts:")
    print(comparison["s03_verdict"].value_counts())
    print("\ns04 verdict counts:")
    print(comparison["s04_verdict"].value_counts())

    return valid, verdicts_after


def dominant_stratum(rows: pd.DataFrame) -> pd.DataFrame:
    counts = rows.groupby(["ambient_temperature", "load_level"]).size()
    dominant_temp, dominant_load = counts.idxmax()
    return rows[(rows["ambient_temperature"] == dominant_temp) & (rows["load_level"] == dominant_load)]


def step3_soh(discharge: pd.DataFrame, valid: pd.DataFrame, verdicts: dict) -> pd.DataFrame:
    section("Step 3: SOH")

    final_ref = discharge.drop_duplicates("battery_id").set_index("battery_id")["capacity_ref_step1"].to_dict()
    for battery_id, verdict in verdicts.items():
        if verdict != "STRATIFIED":
            continue
        battery_valid = valid[valid["battery_id"] == battery_id]
        if battery_valid.empty:
            continue
        stratum_ref = top5_median_capacity(dominant_stratum(battery_valid))
        print(f"{battery_id}: STRATIFIED -- capacity_ref recomputed on dominant stratum: {stratum_ref:.4f} (whole-battery was {final_ref[battery_id]:.4f})")
        final_ref[battery_id] = stratum_ref

    discharge = discharge.copy()
    discharge["capacity_ref"] = discharge["battery_id"].map(final_ref)
    discharge["soh"] = discharge["Capacity_num"] / discharge["capacity_ref"]
    discharge["soh_pct"] = 100 * discharge["soh"]

    print("\nSOH range per battery:")
    print(discharge.groupby("battery_id")["soh"].agg(["min", "max"]).to_string())

    valid_exceeding = discharge[(~discharge["invalid_measurement"]) & (discharge["soh"] > 1.0)]
    print(
        f"\nvalid rows with soh > 1.0: {len(valid_exceeding)} of {int((~discharge['invalid_measurement']).sum())} "
        f"valid rows. This is a MECHANICAL ARTIFACT of capacity_ref = median of the top {CAPACITY_REF_TOP_N} "
        f"valid Capacity values -- by construction the top 2 of those {CAPACITY_REF_TOP_N} values exceed their "
        f"own median, so roughly 2 rows per battery are guaranteed to land above soh=1.0 regardless of any real "
        f"degradation behaviour. Not clipped here, and NOT evidence that early cycles genuinely exceed a "
        f"baseline capacity -- see the capacity_ref sensitivity check below."
    )
    if not valid_exceeding.empty:
        print(valid_exceeding.groupby("battery_id").size().to_string())

    return discharge


def capacity_ref_estimators(rows: pd.DataFrame) -> dict[str, float]:
    positive = rows[rows["Capacity_num"].notna() & (rows["Capacity_num"] > 0)]
    if positive.empty:
        return {"top5_median": float("nan"), "first5_median": float("nan"), "single_max": float("nan")}
    return {
        "top5_median": positive["Capacity_num"].nlargest(CAPACITY_REF_TOP_N).median(),
        "first5_median": positive.sort_values("cycle_index")["Capacity_num"].head(CAPACITY_REF_TOP_N).median(),
        "single_max": positive["Capacity_num"].max(),
    }


def first_crossing_cycle(rows: pd.DataFrame, capacity_ref: float, threshold: float) -> float:
    if pd.isna(capacity_ref) or capacity_ref <= 0:
        return float("nan")
    ordered = rows.sort_values("cycle_index")
    soh = ordered["Capacity_num"] / capacity_ref
    crossed = ordered.loc[soh <= threshold, "cycle_index"]
    return float(crossed.iloc[0]) if not crossed.empty else float("nan")


def step3b_capacity_ref_sensitivity(valid: pd.DataFrame, crossing_threshold: float = 0.80) -> None:
    section("Step 3b: capacity_ref sensitivity (diagnostic only -- does not change the written table)")

    rows = []
    for battery_id, group in valid.groupby("battery_id"):
        estimators = capacity_ref_estimators(group)
        crossings = {name: first_crossing_cycle(group, ref, crossing_threshold) for name, ref in estimators.items()}
        observed = [c for c in crossings.values() if not pd.isna(c)]
        spread = max(observed) - min(observed) if len(observed) >= 2 else float("nan")
        rows.append(
            {
                "battery_id": battery_id,
                "ref_top5_median": estimators["top5_median"],
                "ref_first5_median": estimators["first5_median"],
                "ref_single_max": estimators["single_max"],
                "cross_top5_median": crossings["top5_median"],
                "cross_first5_median": crossings["first5_median"],
                "cross_single_max": crossings["single_max"],
                "spread_cycles": spread,
            }
        )

    table = pd.DataFrame(rows).sort_values("battery_id").reset_index(drop=True)
    print(f"soh={crossing_threshold:.0%} crossing point (cycle_index) by capacity_ref estimator, and the spread across estimators:")
    print(table.to_string(index=False))

    shifted = table[table["spread_cycles"] > 5]
    print(
        f"\nbatteries where the {crossing_threshold:.0%} crossing point shifts by more than 5 cycles "
        f"between estimators: {len(shifted)} of {len(table)} -> {sorted(shifted['battery_id'].tolist())}"
    )
    print(
        "The label boundary (cycle_index at which soh first crosses "
        f"{crossing_threshold:.0%}) is estimator-dependent for these batteries -- the capacity_ref "
        "definition is not yet settled."
    )

    b0045 = table.loc[table["battery_id"] == "B0045"]
    if not b0045.empty:
        ref = b0045.iloc[0]["ref_top5_median"]
        print(
            f"\nB0045 check: capacity_ref={ref:.4f} vs 1.10-2.01 for every other battery. B0045's first "
            "recorded cycle (cycle_index=0) is already only ~1.08 Ah, and its own max Capacity across all "
            "72 recorded discharge cycles never exceeds that value. Its same-batch, same-start-time "
            "siblings (B0046/B0047/B0048, started at the identical timestamp) begin at 1.66-1.73 Ah. So "
            "B0045 simply never shows a fresh/full-capacity reading in the recorded data -- capacity_ref "
            "is real given B0045's own history, but that history starts after this unit had already lost "
            "capacity (or it is an intrinsically weaker cell), not at a comparable fresh baseline."
        )


def plot_soh_vs_cycle(discharge: pd.DataFrame, verdicts: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted_verdicts = set()
    for battery_id, group in discharge.groupby("battery_id"):
        group = group[~group["invalid_measurement"]].dropna(subset=["soh"]).sort_values("cycle_index")
        if group.empty:
            continue
        verdict = verdicts.get(battery_id, "INSUFFICIENT")
        color = VERDICT_COLORS[verdict]
        label = verdict if verdict not in plotted_verdicts else None
        plotted_verdicts.add(verdict)
        ax.plot(group["cycle_index"], group["soh"], color=color, alpha=0.6, linewidth=1, label=label)

    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("cycle_index")
    ax.set_ylabel("soh")
    ax.set_title("NASA: SOH vs cycle_index (valid rows only), all batteries, coloured by verdict")
    ax.legend(title="verdict")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "s04_soh_vs_cycle_all_batteries.png")
    plt.close(fig)


def plot_invalid_flagged(discharge: pd.DataFrame) -> None:
    counts = pd.crosstab(discharge["battery_id"], discharge["invalid_reason"]).drop(columns=["valid"], errors="ignore")
    counts = counts.loc[counts.sum(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(12, 5))
    counts.plot(kind="bar", stacked=True, ax=ax, width=0.8)
    ax.set_xlabel("battery_id")
    ax.set_ylabel("flagged rows")
    ax.set_title("NASA: invalid_measurement rows per battery, by reason")
    ax.legend(title="reason")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "s04_invalid_flagged_per_battery.png")
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    discharge, _ = load_discharge_rows()
    print(f"input shape (metadata discharge rows): {discharge.shape}")

    discharge = step1_validity_filter(discharge)

    verdicts_before = s03_equivalent_verdicts(discharge)
    valid, verdicts_after = step2_rediagnose(discharge, verdicts_before)

    discharge = step3_soh(discharge, valid, verdicts_after)
    discharge["verdict"] = discharge["battery_id"].map(verdicts_after)

    step3b_capacity_ref_sensitivity(valid)

    section("Step 4: write")
    output = discharge.drop(columns=["start_time"]).rename(columns={"_start_dt": "start_time", "Capacity_num": "capacity"})[
        [
            "battery_id",
            "cycle_index",
            "start_time",
            "ambient_temperature",
            "load_level",
            "capacity",
            "capacity_ref",
            "soh",
            "soh_pct",
            "invalid_measurement",
            "verdict",
        ]
    ]
    output.to_parquet(OUTPUT_PATH, index=False)
    print(f"output shape: {output.shape}")
    print(f"wrote {OUTPUT_PATH}")

    section("Step 5: plots")
    plot_soh_vs_cycle(discharge, verdicts_after)
    plot_invalid_flagged(discharge)
    print(f"saved plots to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
