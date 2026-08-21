"""Audit the raw data sources: schema sanity, nulls, and source-specific checks.

Read-only. Prints a validation report to stdout and writes plots to
reports/figures/. Never writes data files.
"""

from __future__ import annotations

import cmath
import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from vq_core.paths import PROJECT_ROOT, RAW_EV_DEGRADATION, RAW_EV_TR, RAW_NASA
from vq_core.schema import DROP_ON_LOAD, RE_RCT_COMPLEX_UIDS, TR_EVENT_FLAG_ORDER, TR_PROBABILITY_BINS

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# Source file uses M/D/YYYY H:MM (no leading zeros) -- inference misreads this
# as day-first for some rows, so the format must be pinned explicitly.
TR_TIMESTAMP_FORMAT = "%m/%d/%Y %H:%M"

TR_SAMPLE_CELL_COUNT = 6

TR_LOW_SIGNAL_THRESHOLD = 0.05

# metadata.csv: Capacity/Re/Rct are structurally null off their applicable
# `type` (verified in s02 Gate C) -- a blanket "any null fails" rule is wrong
# for this source. These are the exact known exceptions where Capacity is
# missing on a discharge row anyway (uid is metadata.csv's stable row key);
# anything outside this list is a genuine structural violation, not noise.
NASA_KNOWN_MISSING_CAPACITY_UIDS = {
    4371, 4373, 4375, 4377,  # B0050
    4391, 4395, 4397, 4399, 4401, 4403, 4407, 4409, 4411, 4413,
    4415, 4419, 4421, 4423, 4425, 4427, 4431, 4433, 4435, 4437, 4439,  # B0052
}

NASA_PRESENCE_RULES = {
    "Capacity": {"present_when_type": "discharge", "known_exception_uids": NASA_KNOWN_MISSING_CAPACITY_UIDS},
    "Re": {"present_when_type": "impedance", "known_exception_uids": set()},
    "Rct": {"present_when_type": "impedance", "known_exception_uids": set()},
}

failures: list[str] = []


def fail(message: str) -> None:
    failures.append(message)
    print(f"[FAIL] {message}")


def section(title: str) -> None:
    print()
    print(f"=== {title} ===")


def report_basic(name: str, df: pd.DataFrame, fail_on_any_null: bool = True) -> None:
    section(f"{name}: shape / dtypes / nulls / duplicates")
    print(f"shape: {df.shape}")
    print("dtypes:")
    print(df.dtypes)

    null_counts = df.isnull().sum()
    print("null counts:")
    print(null_counts)
    total_nulls = int(null_counts.sum())
    if fail_on_any_null:
        if total_nulls:
            fail(f"{name}: {total_nulls} null value(s) found")
    else:
        print(
            f"({total_nulls} null value(s) total -- not auto-failed here; nulls are structurally "
            f"expected for this source, see the structural check below)"
        )

    dup_count = int(df.duplicated().sum())
    print(f"duplicate rows: {dup_count}")

    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        print("numeric min/max:")
        print(numeric.agg(["min", "max"]).T)


def is_present(series: pd.Series) -> pd.Series:
    """Non-missing, treating the literal string '[]' as missing like NaN."""
    return series.notna() & (series.astype(str).str.strip() != "[]")


def investigate_object_dtype_columns(df: pd.DataFrame, columns: list[str]) -> None:
    section("nasa/metadata.csv: why Capacity/Re/Rct load as object, not float")
    for column in columns:
        raw = df[column]
        coerced = pd.to_numeric(raw, errors="coerce")
        offending_mask = coerced.isna() & raw.notna()
        offending_rows = df.loc[offending_mask, ["uid", "battery_id", column]]

        if offending_rows.empty:
            print(f"{column}: every non-null raw value parses as numeric -- object dtype must come from NaN mixing with real values")
            continue

        is_placeholder = offending_rows[column].astype(str).str.strip() == "[]"
        is_known_complex = offending_rows["uid"].isin(RE_RCT_COMPLEX_UIDS)
        placeholder_rows = offending_rows[is_placeholder]
        complex_rows = offending_rows[is_known_complex & ~is_placeholder]
        unknown_rows = offending_rows[~is_placeholder & ~is_known_complex]

        if not placeholder_rows.empty:
            print(
                f"{column}: {len(placeholder_rows)} occurrence(s) of the literal placeholder "
                f"string '[]' -- a legitimate missing marker, not a parsing problem."
            )

        if not complex_rows.empty:
            print(
                f"{column}: {len(complex_rows)} row(s) store a genuine complex value -- impedance "
                f"is physically R + jX, so this is a VALID measurement requiring complex() parsing, "
                f"NOT malformed data (uids listed in vq_core.schema.RE_RCT_COMPLEX_UIDS). "
                f"pd.to_numeric(errors='coerce') cannot parse these and would silently null out a "
                f"present measurement:"
            )
            for _, row in complex_rows.iterrows():
                z = complex(row[column])
                magnitude = abs(z)
                phase_deg = math.degrees(cmath.phase(z))
                print(
                    f"    uid={row['uid']} battery_id={row['battery_id']}: {row[column]!r} -> "
                    f"real={z.real:.6f}, magnitude={magnitude:.6f}, phase={phase_deg:.2f}deg"
                )

        if not unknown_rows.empty:
            print(
                f"[NOTICE] {column}: {len(unknown_rows)} non-numeric token(s) that are NEITHER the "
                f"'[]' placeholder NOR in the known RE_RCT_COMPLEX_UIDS exception list -- "
                f"pd.to_numeric(errors='coerce') would SILENTLY turn these into NaN, and this is not "
                f"a documented exception:"
            )
            for _, row in unknown_rows.iterrows():
                print(f"    uid={row['uid']} battery_id={row['battery_id']}: {row[column]!r}")


def check_nasa_structural_presence(df: pd.DataFrame) -> None:
    section("nasa/metadata.csv: structural presence check (Capacity/Re/Rct vs type)")
    for column, rule in NASA_PRESENCE_RULES.items():
        expected_type = rule["present_when_type"]
        known_exceptions = rule["known_exception_uids"]
        present = is_present(df[column])

        present_elsewhere = df[(df["type"] != expected_type) & present]
        if not present_elsewhere.empty:
            fail(
                f"nasa: {column} present on {len(present_elsewhere)} row(s) with type != "
                f"'{expected_type}' (uid={present_elsewhere['uid'].tolist()})"
            )

        missing_where_expected = df[(df["type"] == expected_type) & ~present]
        known_missing = missing_where_expected[missing_where_expected["uid"].isin(known_exceptions)]
        unknown_missing = missing_where_expected[~missing_where_expected["uid"].isin(known_exceptions)]

        if not unknown_missing.empty:
            fail(
                f"nasa: {column} missing on {len(unknown_missing)} '{expected_type}' row(s) not in "
                f"the known-exception list (uid={unknown_missing['uid'].tolist()})"
            )

        if not known_missing.empty:
            batteries = sorted(known_missing["battery_id"].unique().tolist())
            print(
                f"{column}: {len(known_missing)} '{expected_type}' row(s) missing -- KNOWN EXCEPTION "
                f"(battery_id={batteries}), not a failure."
            )

        stale = known_exceptions - set(missing_where_expected["uid"])
        if stale:
            print(f"[WARN] {column}: known-exception uid(s) {sorted(stale)} are no longer missing -- exception list may be stale.")

        if present_elsewhere.empty and unknown_missing.empty:
            print(f"{column}: structural pattern OK ({int(present.sum())} present, all on '{expected_type}' rows).")


def check_drop_on_load(name: str, df: pd.DataFrame, source_key: str) -> None:
    columns = DROP_ON_LOAD.get(source_key)
    if columns is None:
        return
    missing = [c for c in columns if c not in df.columns]
    if missing:
        fail(f"{name}: DROP_ON_LOAD columns missing from source: {missing}")
    else:
        print(f"DROP_ON_LOAD check: all {len(columns)} columns present {columns}")


def expected_tr_probability_range(flag: str) -> tuple[float, float]:
    """Expected [lo, hi] TR_Probability bounds for `flag` per TR_PROBABILITY_BINS."""
    index = TR_EVENT_FLAG_ORDER.index(flag)
    lo = 0.0 if index == 0 else TR_PROBABILITY_BINS[index - 1]
    hi = 1.0 if index == len(TR_EVENT_FLAG_ORDER) - 1 else TR_PROBABILITY_BINS[index]
    return lo, hi


def audit_nasa() -> None:
    section("NASA raw source")
    files = sorted(RAW_NASA.glob("*.csv")) if RAW_NASA.is_dir() else []
    if not files:
        print(f"no files found in {RAW_NASA} -- skipping (source is currently empty)")
        return
    for path in files:
        df = pd.read_csv(path)
        if path.name == "metadata.csv":
            # Capacity/Re/Rct are structurally null off their applicable `type`
            # (verified in s02 Gate C) -- don't auto-fail on that, check the
            # actual structural pattern instead.
            report_basic(f"nasa/{path.name}", df, fail_on_any_null=False)
            investigate_object_dtype_columns(df, ["Capacity", "Re", "Rct"])
            check_nasa_structural_presence(df)
        else:
            report_basic(f"nasa/{path.name}", df)


def audit_ev_degradation() -> pd.DataFrame:
    section("EV degradation raw source")
    df = pd.read_csv(RAW_EV_DEGRADATION)
    report_basic("ev_degradation", df)
    check_drop_on_load("ev_degradation", df, "ev_degradation")

    section("ev_degradation: Battery_Status value counts")
    print(df["Battery_Status"].value_counts())

    section("ev_degradation: SoH_Percent distribution by Battery_Status")
    print(df.groupby("Battery_Status")["SoH_Percent"].describe())

    positive = df["Battery_Status"] != "Healthy"
    positive_fraction = positive.mean()
    print(f"positive-class fraction (Battery_Status != 'Healthy'): {positive_fraction:.4%}")
    if positive_fraction < 0.01:
        print(f"[WARN] positive-class fraction {positive_fraction:.4%} is below 1%")

    return df


def audit_ev_charging_tr() -> pd.DataFrame:
    section("EV charging TR raw source")
    df = pd.read_csv(RAW_EV_TR)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format=TR_TIMESTAMP_FORMAT)
    report_basic("ev_charging_tr", df)
    check_drop_on_load("ev_charging_tr", df, "ev_charging_tr")

    section("ev_charging_tr: chronological order per CellID")
    unsorted_cells = [
        cell_id
        for cell_id, group in df.groupby("CellID", sort=False)
        if not group["Timestamp"].is_monotonic_increasing
    ]
    if unsorted_cells:
        fail(f"ev_charging_tr: timestamps not sorted per CellID for: {unsorted_cells}")
    else:
        print("all cells have chronologically sorted timestamps")

    section("ev_charging_tr: rows-per-cell distribution")
    rows_per_cell = df.groupby("CellID", sort=False).size()
    print(rows_per_cell.describe())

    section("ev_charging_tr: EventFlag x BMS_Status cross-tab")
    print(pd.crosstab(df["EventFlag"], df["BMS_Status"]))

    section("ev_charging_tr: TR_Probability vs EventFlag correlation")
    event_flag_code = pd.Categorical(
        df["EventFlag"], categories=TR_EVENT_FLAG_ORDER, ordered=True
    ).codes
    correlation = pd.Series(event_flag_code).corr(df["TR_Probability"])
    print(f"EventFlag order: {TR_EVENT_FLAG_ORDER}")
    print(f"corr(TR_Probability, EventFlag as ordinal): {correlation:.4f}")

    section("ev_charging_tr: verify EventFlag is a deterministic TR_Probability binning")
    observed_ranges = df.groupby("EventFlag")["TR_Probability"].agg(["min", "max"])
    for flag in TR_EVENT_FLAG_ORDER:
        lo, hi = expected_tr_probability_range(flag)
        obs_min, obs_max = observed_ranges.loc[flag, "min"], observed_ranges.loc[flag, "max"]
        if obs_min < lo or obs_max > hi:
            fail(
                f"ev_charging_tr: {flag} observed TR_Probability range "
                f"[{obs_min}, {obs_max}] falls outside expected [{lo}, {hi}]"
            )
        else:
            print(
                f"{flag}: observed [{obs_min:.3f}, {obs_max:.3f}] "
                f"within expected [{lo:.3f}, {hi:.3f}]"
            )

    for lower, upper in zip(TR_EVENT_FLAG_ORDER, TR_EVENT_FLAG_ORDER[1:]):
        lower_max = observed_ranges.loc[lower, "max"]
        upper_min = observed_ranges.loc[upper, "min"]
        if lower_max >= upper_min:
            fail(
                f"ev_charging_tr: {lower} (max {lower_max}) overlaps "
                f"{upper} (min {upper_min})"
            )
        else:
            print(f"{lower} / {upper}: no overlap ({lower_max:.3f} < {upper_min:.3f})")

    section("ev_charging_tr: TR_Probability correlation with every numeric column")
    numeric_columns = df.select_dtypes(include="number").columns.drop("TR_Probability")
    correlations = (
        df[numeric_columns].corrwith(df["TR_Probability"]).sort_values(key=lambda s: s.abs(), ascending=False)
    )
    for column, r in correlations.items():
        marker = " [low-signal]" if abs(r) < TR_LOW_SIGNAL_THRESHOLD else ""
        print(f"  {column}: {r:.4f}{marker}")

    return df


def plot_soh_histogram_by_status(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for status, group in df.groupby("Battery_Status"):
        ax.hist(group["SoH_Percent"], bins=30, alpha=0.6, label=status)
    ax.set_xlabel("SoH_Percent")
    ax.set_ylabel("count")
    ax.set_title("SoH distribution by Battery_Status")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "ev_degradation_soh_by_status.png")
    plt.close(fig)


def plot_tr_temp_vs_soh(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for flag, group in df.groupby("EventFlag"):
        ax.scatter(group["StateOfHealth_%"], group["MaxTemp_C"], alpha=0.6, label=flag, s=20)
    ax.set_xlabel("StateOfHealth_%")
    ax.set_ylabel("MaxTemp_C")
    ax.set_title("MaxTemp_C vs StateOfHealth_% by EventFlag")
    ax.legend(title="EventFlag")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "ev_charging_tr_maxtemp_vs_soh.png")
    plt.close(fig)


def plot_tr_temp_over_time_by_cell(df: pd.DataFrame) -> None:
    sample_cells = (
        df.groupby("CellID", sort=False)
        .size()
        .sort_values(ascending=False)
        .head(TR_SAMPLE_CELL_COUNT)
        .index
    )
    print(f"per-cell temperature plot sample cells: {list(sample_cells)}")

    fig, ax = plt.subplots(figsize=(9, 5))
    for cell_id in sample_cells:
        group = df[df["CellID"] == cell_id].sort_values("Timestamp")
        ax.plot(group["Timestamp"], group["MaxTemp_C"], marker="o", label=cell_id)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("MaxTemp_C")
    ax.set_title(f"MaxTemp_C over time for {TR_SAMPLE_CELL_COUNT} sampled cells")
    ax.legend(title="CellID")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "ev_charging_tr_temp_over_time_sample_cells.png")
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    audit_nasa()
    degradation_df = audit_ev_degradation()
    tr_df = audit_ev_charging_tr()

    plot_soh_histogram_by_status(degradation_df)
    plot_tr_temp_vs_soh(tr_df)
    plot_tr_temp_over_time_by_cell(tr_df)

    section("RESULT")
    if failures:
        print("FAIL")
        for message in failures:
            print(f"  - {message}")
        raise SystemExit(1)
    print("PASS")


if __name__ == "__main__":
    main()
