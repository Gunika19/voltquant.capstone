"""Build the per-discharge-cycle NASA table with SOH ground truth.

Writes data/02_intermediate/nasa_discharge_cycles.parquet.

SOH definition: soh = capacity / capacity_ref, where capacity_ref is the
median of a battery's own top-5 highest valid Capacity readings (its
beginning-of-life capacity). This is relative to the CELL'S OWN start-of-life
capacity, per standard battery-prognostics convention -- not to the 2 Ah
nameplate rating. NOMINAL_CAPACITY_AH exists only for the sensitivity check
below: fresh cells in this batch average 0.884 of nominal with std 0.051, so
using nominal as the reference would encode manufacturing spread as if it
were degradation.

B0045 has no fresh-state reading anywhere in its recorded history (see
vq_core.schema.FRESH_STATE_MISSING) and gets verdict NO_BASELINE with
capacity_ref/soh/soh_pct left null -- its rows are kept, not dropped.

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
from vq_core.schema import FRESH_STATE_MISSING, NOMINAL_CAPACITY_AH

from vq_data.s02_verify_nasa import (
    METADATA_SEARCH_DEPTH,
    find_metadata_csv,
    find_per_cycle_dir,
    parse_start_time,
)
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
CROSSING_THRESHOLD = 0.80

# All Spearman/crossing-point work uses discharge_cycle_index, not
# record_index -- see load_discharge_rows().
CYCLE_COLUMN = "discharge_cycle_index"

# Checkpoint from before B0045 got NO_BASELINE treatment: the discharge_cycle_index
# fix (change 1) must not itself move any battery's verdict, since Spearman
# correlation only depends on rank order, and discharge_cycle_index preserves
# the same relative order among discharge rows as the old cross-type index.
EXPECTED_VERDICT_COUNTS_AFTER_INDEX_FIX = {"CLEAN": 23, "STRATIFIED": 1, "ANOMALOUS": 8, "INSUFFICIENT": 2}

ESTIMATOR_NAMES = ["top5_median", "first5_median", "single_max", "nominal"]

# A battery's first-ever discharge reading is treated as "capturing a fresh
# state" if it's close to the documented 2 Ah rating. The real data splits
# cleanly here: 18 batteries land at 0.83-1.02 of nominal on their very first
# discharge, then there's a gap down to B0045 at 0.54 and a noisy tail below
# that (down to 0.03 -- clearly glitches, not partially-used cells). 0.80
# sits in that gap.
FRESH_START_MIN_FRACTION_OF_NOMINAL = 0.80

VERDICT_COLORS = {
    "CLEAN": "tab:green",
    "STRATIFIED": "tab:orange",
    "ANOMALOUS": "tab:red",
    "INSUFFICIENT": "tab:gray",
    "NO_BASELINE": "tab:purple",
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
    # record_index spans EVERY measurement type (charge, discharge, impedance)
    # -- it's each row's chronological position among all of a battery's
    # recorded tests, not a discharge-cycle counter. Kept only for provenance;
    # nothing downstream should use it to order or correlate discharge cycles.
    df["record_index"] = df.groupby("battery_id")["_start_dt"].rank(method="first").astype(int) - 1
    df["Capacity_num"] = pd.to_numeric(df["Capacity"], errors="coerce")

    discharge = df[df["type"] == "discharge"].copy()
    # discharge_cycle_index: zero-based, contiguous rank of start_time within
    # battery_id over discharge rows ONLY. This is the actual discharge-cycle
    # counter and is what every correlation/plot/crossing-point below uses.
    discharge["discharge_cycle_index"] = (
        discharge.groupby("battery_id")["_start_dt"].rank(method="first").astype(int) - 1
    )

    sample_filename = next(iter(discharge["filename"].dropna()))
    per_cycle_dir = find_per_cycle_dir(metadata_path.parent, sample_filename)
    if per_cycle_dir is None:
        raise FileNotFoundError(f"could not locate the per-cycle data directory (looked for {sample_filename})")

    discharge = extract_load_current(discharge, per_cycle_dir)
    discharge["load_level"] = discharge["load_current"].round(1)
    return discharge, per_cycle_dir


def capacity_ref_by_battery(discharge: pd.DataFrame) -> pd.Series:
    """The one, canonical, pre-filter capacity_ref: median of the top-5 highest
    Capacity values among ALL rows with Capacity > 0 for that battery. Must
    always be called on unfiltered data -- computing it from `valid` (i.e.
    after Step 1's own rule has excluded rows judged against this same
    reference) is exactly the circularity this function exists to avoid."""
    positive = discharge[discharge["Capacity_num"].notna() & (discharge["Capacity_num"] > 0)]
    ref = positive.groupby("battery_id")["Capacity_num"].apply(lambda s: s.nlargest(CAPACITY_REF_TOP_N).median())
    ref.name = "capacity_ref"
    return ref


def step1_validity_filter(discharge: pd.DataFrame) -> pd.DataFrame:
    section("Step 1: validity filter")
    discharge = discharge.copy()

    capacity_ref = capacity_ref_by_battery(discharge)
    print(f"capacity_ref per battery (median of top {CAPACITY_REF_TOP_N} valid Capacity values, computed pre-filter):")
    print(capacity_ref.sort_values().to_string())

    discharge = discharge.merge(capacity_ref, how="left", left_on="battery_id", right_index=True)

    conditions = [
        discharge["Capacity_num"].isna(),
        discharge["Capacity_num"] <= 0,
        discharge["Capacity_num"] < LOW_RATIO * discharge["capacity_ref"],
        discharge["Capacity_num"] > HIGH_RATIO * discharge["capacity_ref"],
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
    pooled = pooled_spearman(s03_discharge, cycle_column=CYCLE_COLUMN)

    h1_strata, _ = stratified_spearman(s03_discharge, ["battery_id", "ambient_temperature"], cycle_column=CYCLE_COLUMN)

    load_variation = s03_discharge.groupby("battery_id")["load_level"].nunique()
    varying_batteries = set(load_variation[load_variation > 1].index)
    h2_strata, _ = stratified_spearman(
        s03_discharge,
        ["battery_id", "ambient_temperature", "load_level"],
        eligible_batteries=varying_batteries,
        cycle_column=CYCLE_COLUMN,
    )

    return classify(pooled, h1_strata, h2_strata)


def step2_rediagnose(discharge: pd.DataFrame, verdicts_before: dict) -> tuple[pd.DataFrame, dict]:
    section("Step 2: re-diagnose (valid rows only)")

    valid = discharge[~discharge["invalid_measurement"]].copy()
    valid_counts = valid.groupby("battery_id").size()

    all_battery_ids = discharge["battery_id"].unique()
    insufficient = sorted(
        set(valid_counts[valid_counts < MIN_VALID_CYCLES].index) | {b for b in all_battery_ids if b not in valid_counts.index}
    )
    print(f"batteries excluded as INSUFFICIENT (< {MIN_VALID_CYCLES} valid discharge cycles): {insufficient}")

    analyzable = valid[~valid["battery_id"].isin(insufficient)]
    pooled_after = pooled_spearman(analyzable, cycle_column=CYCLE_COLUMN)
    strata_after, strata_df = stratified_spearman(
        analyzable,
        ["battery_id", "ambient_temperature", "load_level"],
        min_cycles=STEP2_MIN_STRATUM_CYCLES,
        cycle_column=CYCLE_COLUMN,
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

    section("Step 2: verdict-count checkpoint (discharge_cycle_index fix should not move this)")
    actual_counts = comparison["s04_verdict"].value_counts().to_dict()
    if actual_counts == EXPECTED_VERDICT_COUNTS_AFTER_INDEX_FIX:
        print(f"CONFIRMED: verdict counts unchanged after the discharge_cycle_index fix: {actual_counts}")
    else:
        print(
            f"[DIFFERS] expected {EXPECTED_VERDICT_COUNTS_AFTER_INDEX_FIX}, got {actual_counts} -- "
            "the discharge_cycle_index fix changed at least one battery's verdict; investigate."
        )

    return valid, verdicts_after


def apply_no_baseline_override(verdicts: dict) -> dict:
    verdicts = dict(verdicts)
    for battery_id in FRESH_STATE_MISSING:
        previous = verdicts.get(battery_id)
        verdicts[battery_id] = "NO_BASELINE"
        print(f"{battery_id}: verdict overridden to NO_BASELINE (was {previous}) -- no fresh-state reading exists, per vq_core.schema.FRESH_STATE_MISSING")
    return verdicts


def step3_soh(discharge: pd.DataFrame, verdicts: dict) -> pd.DataFrame:
    section("Step 3: SOH (soh = capacity / capacity_ref, capacity_ref = own top-5 beginning-of-life median)")

    discharge = discharge.copy()
    discharge["soh"] = discharge["Capacity_num"] / discharge["capacity_ref"]
    discharge["soh_pct"] = 100 * discharge["soh"]

    for battery_id in FRESH_STATE_MISSING:
        mask = discharge["battery_id"] == battery_id
        discharge.loc[mask, ["capacity_ref", "soh", "soh_pct"]] = float("nan")
        print(f"{battery_id}: capacity_ref/soh/soh_pct set to null (NO_BASELINE) -- rows kept, not dropped")

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


def capacity_ref_estimators(raw_rows: pd.DataFrame) -> dict[str, float]:
    """The four candidate references, ALWAYS computed from a battery's full,
    unfiltered rows (never from `valid`) -- see capacity_ref_by_battery."""
    positive = raw_rows[raw_rows["Capacity_num"].notna() & (raw_rows["Capacity_num"] > 0)]
    if positive.empty:
        top5_median = first5_median = single_max = float("nan")
    else:
        top5_median = positive["Capacity_num"].nlargest(CAPACITY_REF_TOP_N).median()
        first5_median = positive.sort_values(CYCLE_COLUMN)["Capacity_num"].head(CAPACITY_REF_TOP_N).median()
        single_max = positive["Capacity_num"].max()
    return {
        "top5_median": top5_median,
        "first5_median": first5_median,
        "single_max": single_max,
        "nominal": NOMINAL_CAPACITY_AH,
    }


def first_crossing_cycle(rows: pd.DataFrame, capacity_ref: float, threshold: float) -> float:
    if pd.isna(capacity_ref) or capacity_ref <= 0 or rows.empty:
        return float("nan")
    ordered = rows.sort_values(CYCLE_COLUMN)
    soh = ordered["Capacity_num"] / capacity_ref
    crossed = ordered.loc[soh <= threshold, CYCLE_COLUMN]
    return float(crossed.iloc[0]) if not crossed.empty else float("nan")


def step3b_capacity_ref_sensitivity(discharge: pd.DataFrame, valid: pd.DataFrame) -> dict:
    section("Step 3b: capacity_ref sensitivity (documented limitation -- does not change the written table)")

    # Whether a battery's first-ever discharge reading looks fresh must be
    # judged from the FULL (unfiltered) history, not `valid`: Step1's own
    # "above 110% of ref" rule can exclude exactly that first reading for a
    # battery whose later capacity mostly sits lower (see below) -- using
    # `valid` here would make the check circular.
    raw_positive = discharge[discharge["Capacity_num"].notna() & (discharge["Capacity_num"] > 0)]
    first_ever = raw_positive.sort_values(CYCLE_COLUMN).groupby("battery_id").first()[
        [CYCLE_COLUMN, "Capacity_num", "invalid_measurement"]
    ]

    # NOT dict(some_groupby): DataFrameGroupBy has a `.keys` ATTRIBUTE (the
    # grouping column name, a plain string) that dict()'s mapping duck-typing
    # mistakes for a `.keys()` method, raising "'str' object is not callable".
    raw_groups = {name: group for name, group in discharge.groupby("battery_id")}
    valid_groups = {name: group for name, group in valid.groupby("battery_id")}

    rows = []
    fresh_cycle0_soh_by_estimator: dict[str, list[float]] = {name: [] for name in ESTIMATOR_NAMES}
    fresh_batteries: list[str] = []
    fresh_excluded_by_step1: list[str] = []

    for battery_id, raw_group in sorted(raw_groups.items()):
        valid_group = valid_groups.get(battery_id, raw_group.iloc[0:0])
        estimators = capacity_ref_estimators(raw_group)

        first_capacity = first_ever.loc[battery_id, "Capacity_num"]
        is_fresh_start = first_capacity >= FRESH_START_MIN_FRACTION_OF_NOMINAL * NOMINAL_CAPACITY_AH
        if is_fresh_start:
            fresh_batteries.append(battery_id)
            if bool(first_ever.loc[battery_id, "invalid_measurement"]):
                fresh_excluded_by_step1.append(battery_id)

        row = {"battery_id": battery_id, "fresh_start": is_fresh_start}
        crossings = {}
        for name in ESTIMATOR_NAMES:
            ref = estimators[name]
            crossings[name] = first_crossing_cycle(valid_group, ref, CROSSING_THRESHOLD)
            soh = (
                valid_group.sort_values(CYCLE_COLUMN)["Capacity_num"] / ref
                if not pd.isna(ref) and ref > 0 and not valid_group.empty
                else pd.Series(dtype=float)
            )

            row[f"ref_{name}"] = ref
            row[f"cross_{name}"] = crossings[name]
            row[f"soh_min_{name}"] = soh.min() if not soh.empty else float("nan")
            row[f"soh_max_{name}"] = soh.max() if not soh.empty else float("nan")

            if is_fresh_start:
                fresh_soh0 = first_capacity / ref if not pd.isna(ref) and ref > 0 else float("nan")
                fresh_cycle0_soh_by_estimator[name].append(fresh_soh0)

        observed_crossings = [c for c in crossings.values() if not pd.isna(c)]
        row["spread_cycles"] = max(observed_crossings) - min(observed_crossings) if len(observed_crossings) >= 2 else float("nan")
        rows.append(row)

    table = pd.DataFrame(rows).sort_values("battery_id").reset_index(drop=True)

    print(f"soh={CROSSING_THRESHOLD:.0%} crossing point (discharge_cycle_index) by capacity_ref estimator:")
    print(table[["battery_id"] + [f"cross_{n}" for n in ESTIMATOR_NAMES] + ["spread_cycles"]].to_string(index=False))

    print("\ncapacity_ref value by estimator (all computed from unfiltered rows -- consistent with Step 1's own value for top5_median):")
    print(table[["battery_id"] + [f"ref_{n}" for n in ESTIMATOR_NAMES]].to_string(index=False))

    shifted = table[table["spread_cycles"] > 5]
    shifted_batteries = sorted(shifted["battery_id"].tolist())
    print(
        f"\nbatteries where the {CROSSING_THRESHOLD:.0%} crossing point shifts by more than 5 cycles "
        f"across the four estimators: {len(shifted)} of {len(table)} -> {shifted_batteries}"
    )

    section("Step 3b: re-checking the Step1-filter/estimator tension after computing refs pre-filter")
    fresh_batteries = sorted(fresh_batteries)
    print(
        f"batteries treated as capturing a fresh state (first-ever discharge Capacity, from the "
        f"UNFILTERED history, >= {FRESH_START_MIN_FRACTION_OF_NOMINAL:.0%} of nominal): "
        f"{len(fresh_batteries)} of {len(table)} -> {fresh_batteries}"
    )
    if fresh_excluded_by_step1:
        print(
            f"[NOTE] {len(fresh_excluded_by_step1)} of those fresh-start batteries "
            f"({sorted(fresh_excluded_by_step1)}) STILL have their own fresh first reading flagged "
            f"invalid_measurement by Step1 ('above_110pct_of_ref'), even now that capacity_ref is "
            f"computed consistently pre-filter everywhere. This tension does NOT resolve from fixing "
            f"the estimator circularity: Step1's own capacity_ref was already computed pre-filter "
            f"before this change too (it was only the diagnostic estimators here that were recomputed "
            f"from `valid` and disagreed with it). The 110% rule itself is what excludes these rows: "
            f"a battery's first reading naturally sits above the median of ITS OWN top-5 (which "
            f"includes that very reading), so a fresh first cycle can legitimately be its battery's "
            f"single highest value while still exceeding 110% of the top-5 median. No special-casing "
            f"is applied for this beyond B0045: unlike B0045, these batteries' capacity_ref values "
            f"(1.48-1.52) sit well within the normal cross-battery range (1.10-2.01) -- only their "
            f"single fresh row is excluded from `valid`, not their usability as a baseline."
        )
    else:
        print("RESOLVED: no fresh-start battery has its first reading excluded by Step1 any more.")

    print(f"\nsoh at first-ever reading, restricted to those {len(fresh_batteries)} fresh-start batteries, by estimator:")
    for name in ESTIMATOR_NAMES:
        values = fresh_cycle0_soh_by_estimator[name]
        if values:
            series = pd.Series(values)
            print(f"  {name}: mean={series.mean():.4f} std={series.std():.4f} min={series.min():.4f} max={series.max():.4f} (n={len(values)})")
        else:
            print(f"  {name}: no fresh-start batteries had a computable ratio")

    section("Step 3b: documented limitation -- top5_median (operative) vs nominal")
    nominal_fresh = fresh_cycle0_soh_by_estimator["nominal"]
    if nominal_fresh:
        nominal_series = pd.Series(nominal_fresh)
        top5_series = pd.Series(fresh_cycle0_soh_by_estimator["top5_median"])
        print(
            f"top5_median (OPERATIVE): mean={top5_series.mean():.4f} std={top5_series.std():.4f} at first reading "
            f"for the fresh-start cohort -- tautologically near 1.0, since it's fit from each battery's own "
            f"highest readings. That's exactly why it's the right choice: it measures fade relative to what "
            f"THIS cell could actually deliver fresh, per standard prognostics convention."
        )
        print(
            f"nominal (NOT operative): mean={nominal_series.mean():.4f} std={nominal_series.std():.4f} -- "
            f"~12% below 1.0 on average. A fixed 2 Ah reference would encode that gap as if every fresh cell "
            f"had already lost ~12% capacity, when it's manufacturing/test-protocol spread across cells rated "
            f"nominally identical. Kept in this table only as a documented limitation and cross-check, not as "
            f"the operative reference."
        )
    else:
        print("No fresh-start batteries were found -- cannot report the nominal comparison.")

    return {"shifted_more_than_5_cycles": len(shifted), "shifted_batteries": shifted_batteries}


def plot_soh_vs_cycle(discharge: pd.DataFrame, verdicts: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted_verdicts = set()
    for battery_id, group in discharge.groupby("battery_id"):
        group = group[~group["invalid_measurement"]].dropna(subset=["soh"]).sort_values(CYCLE_COLUMN)
        if group.empty:
            continue
        verdict = verdicts.get(battery_id, "INSUFFICIENT")
        color = VERDICT_COLORS[verdict]
        label = verdict if verdict not in plotted_verdicts else None
        plotted_verdicts.add(verdict)
        ax.plot(group[CYCLE_COLUMN], group["soh"], color=color, alpha=0.6, linewidth=1, label=label)

    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("discharge_cycle_index")
    ax.set_ylabel("soh")
    ax.set_title("NASA: SOH vs discharge_cycle_index (valid rows only), all batteries, coloured by verdict")
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

    verdicts_after = apply_no_baseline_override(verdicts_after)

    discharge = step3_soh(discharge, verdicts_after)
    discharge["verdict"] = discharge["battery_id"].map(verdicts_after)

    sensitivity_stats = step3b_capacity_ref_sensitivity(discharge, valid)

    section("Step 4: write")
    output = discharge.drop(columns=["start_time"]).rename(columns={"_start_dt": "start_time", "Capacity_num": "capacity"})[
        [
            "battery_id",
            "discharge_cycle_index",
            "record_index",
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

    section("FINAL SUMMARY")
    verdict_counts = pd.Series(verdicts_after).value_counts()
    print("battery count by verdict:")
    print(verdict_counts.to_string())
    usable = sum(count for label, count in verdict_counts.items() if label not in ("INSUFFICIENT", "NO_BASELINE"))
    print(f"\nusable cells (have a computed SOH baseline -- excludes INSUFFICIENT and NO_BASELINE): {usable} of {len(verdicts_after)}")
    print(
        f"batteries whose {CROSSING_THRESHOLD:.0%} crossing shifts by more than 5 cycles across the "
        f"four capacity_ref estimators: {sensitivity_stats['shifted_more_than_5_cycles']} of {len(verdicts_after)} "
        f"-> {sensitivity_stats['shifted_batteries']}"
    )


if __name__ == "__main__":
    main()
