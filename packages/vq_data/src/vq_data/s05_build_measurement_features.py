"""Build measurement-level features from the NASA per-cycle discharge CSVs.

Writes data/03_features/nasa_discharge_measurements.parquet.

Scope: only discharge cycles from batteries with verdict CLEAN or STRATIFIED
in nasa_discharge_cycles.parquet, and only rows where invalid_measurement is
False (see s04_build_discharge_table.py).

`soh` is carried on every row as a LABEL column, not a feature -- it exists
here so downstream code can join measurements to their target without a
second lookup. Any feature matrix built from this table must drop `soh`
first; vq_core.guards.assert_no_label_leakage will raise if it doesn't (it
matches `soh` case-insensitively via vq_core.schema.LABEL_ONLY_FEATURES).
Do not call it on *this* table -- same convention as s04's output.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from vq_core.paths import PROJECT_ROOT

from vq_data.s04_build_discharge_table import load_discharge_rows

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
LABELS_PATH = PROJECT_ROOT / "data" / "02_intermediate" / "nasa_discharge_cycles.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "03_features" / "nasa_discharge_measurements.parquet"

SCOPE_VERDICTS = ["CLEAN", "STRATIFIED"]
SHORT_CYCLE_MIN_SAMPLES = 20
ROLLING_WINDOW_SAMPLES = 5

# Sample-count windows span different PHYSICAL durations depending on
# sampling rate (9.3-20.5s, including within a single battery's own history
# for B0005/6/7 -- see s05b_check_sampling_confound.py): 5 samples is 46.5s
# at the fast rate but 102.5s at the slow one, a 2.2x difference in what's
# supposedly the "same" smoothing window. 60s is chosen instead because it
# stays >= 3 samples even at the slowest observed rate (60/20.5 -> 3 points,
# enough for a defined std) while remaining short relative to a typical
# cycle (order of 1000-4000s), so it smooths locally rather than averaging
# over a large fraction of the discharge curve.
ROLLING_WINDOW_SECONDS = 60

DT_DT_IMPLAUSIBLE_ABS = 5.0
VOLTAGE_PLAUSIBLE_RANGE = (2.0, 4.5)
TEMPERATURE_PLAUSIBLE_RANGE = (0.0, 100.0)

RAW_MEASUREMENT_COLUMNS = [
    "Voltage_measured",
    "Current_measured",
    "Temperature_measured",
    "Current_load",
    "Voltage_load",
]

ENGINEERED_FEATURE_COLUMNS = [
    "dT_dt",
    "dV_dt",
    "dI_dt",
    "power",
    "roll_mean_temperature_samples",
    "roll_std_temperature_samples",
    "roll_mean_voltage_samples",
    "roll_std_voltage_samples",
    "roll_mean_temperature_60s",
    "roll_std_temperature_60s",
    "roll_mean_voltage_60s",
    "roll_std_voltage_60s",
    "temp_rise",
    "sampling_interval_s",
]


def section(title: str) -> None:
    print()
    print(f"=== {title} ===")


def load_scope() -> tuple[pd.DataFrame, Path]:
    section("Scope")
    labels = pd.read_parquet(LABELS_PATH)
    qualifying = labels[labels["verdict"].isin(SCOPE_VERDICTS) & (~labels["invalid_measurement"])].copy()

    cells = sorted(qualifying["battery_id"].unique())
    print(f"cells in scope (verdict in {SCOPE_VERDICTS}, invalid_measurement False): {len(cells)}")
    print(cells)
    print(f"\nexpected discharge cycles to load: {len(qualifying)}")
    print(qualifying.groupby("battery_id").size().to_string())

    discharge, per_cycle_dir = load_discharge_rows()
    qualifying = qualifying.merge(
        discharge[["battery_id", "discharge_cycle_index", "filename"]],
        on=["battery_id", "discharge_cycle_index"],
        how="left",
    )
    missing = int(qualifying["filename"].isna().sum())
    if missing:
        raise AssertionError(f"{missing} qualifying cycle(s) have no matching per-cycle filename -- labels and metadata have drifted")

    return qualifying, per_cycle_dir


def load_measurements(qualifying: pd.DataFrame, per_cycle_dir: Path) -> pd.DataFrame:
    section("Load")
    start = time.time()
    frames = []
    short_cycles = []
    for row in qualifying.itertuples(index=False):
        cycle_df = pd.read_csv(per_cycle_dir / row.filename)
        is_short = len(cycle_df) < SHORT_CYCLE_MIN_SAMPLES
        if is_short:
            short_cycles.append((row.battery_id, row.discharge_cycle_index, len(cycle_df)))

        cycle_df["battery_id"] = row.battery_id
        cycle_df["discharge_cycle_index"] = row.discharge_cycle_index
        cycle_df["ambient_temperature"] = row.ambient_temperature
        cycle_df["load_level"] = row.load_level
        cycle_df["soh"] = row.soh
        cycle_df["short_cycle"] = is_short
        frames.append(cycle_df)

    measurements = pd.concat(frames, ignore_index=True)
    elapsed = time.time() - start

    print(f"loaded {len(qualifying)} per-cycle files in {elapsed:.1f}s -> {len(measurements)} measurement rows")
    print(f"cycles flagged short_cycle (< {SHORT_CYCLE_MIN_SAMPLES} samples): {len(short_cycles)}")
    if short_cycles:
        for battery_id, cycle_index, n_samples in short_cycles:
            print(f"  {battery_id} cycle {cycle_index}: {n_samples} samples")

    return measurements


def compute_features(measurements: pd.DataFrame) -> pd.DataFrame:
    section("Features")
    measurements = measurements.sort_values(["battery_id", "discharge_cycle_index", "Time"]).reset_index(drop=True)
    grouped = measurements.groupby(["battery_id", "discharge_cycle_index"], sort=False)

    dt = grouped["Time"].diff()
    section("sampling-interval distribution (seconds between consecutive samples, within a cycle)")
    print(dt.describe().to_string())
    non_positive = int((dt <= 0).sum())
    if non_positive:
        print(f"[NOTE] {non_positive} non-positive interval(s) found (duplicate or out-of-order Time) -- these will produce inf/-inf derivatives, not dropped")

    measurements["dT_dt"] = grouped["Temperature_measured"].diff() / dt
    measurements["dV_dt"] = grouped["Voltage_measured"].diff() / dt
    measurements["dI_dt"] = grouped["Current_measured"].diff() / dt
    measurements["power"] = measurements["Voltage_measured"] * measurements["Current_measured"]
    measurements["sampling_interval_s"] = dt

    measurements["roll_mean_temperature_samples"] = grouped["Temperature_measured"].transform(
        lambda s: s.rolling(ROLLING_WINDOW_SAMPLES, min_periods=1).mean()
    )
    measurements["roll_std_temperature_samples"] = grouped["Temperature_measured"].transform(
        lambda s: s.rolling(ROLLING_WINDOW_SAMPLES, min_periods=1).std()
    ).fillna(0.0)
    measurements["roll_mean_voltage_samples"] = grouped["Voltage_measured"].transform(
        lambda s: s.rolling(ROLLING_WINDOW_SAMPLES, min_periods=1).mean()
    )
    measurements["roll_std_voltage_samples"] = grouped["Voltage_measured"].transform(
        lambda s: s.rolling(ROLLING_WINDOW_SAMPLES, min_periods=1).std()
    ).fillna(0.0)

    # Time-based rolling requires a Timedelta/Datetime index -- pandas can't
    # do offset-based windows on a plain float column. Row order is preserved
    # by groupby+rolling here (verified against the original Time values),
    # so assigning back by position is safe.
    time_indexed = measurements.set_index(pd.to_timedelta(measurements["Time"], unit="s"))
    time_grouped = time_indexed.groupby(["battery_id", "discharge_cycle_index"], sort=False)
    window = f"{ROLLING_WINDOW_SECONDS}s"
    measurements["roll_mean_temperature_60s"] = time_grouped["Temperature_measured"].rolling(window).mean().to_numpy()
    measurements["roll_std_temperature_60s"] = (
        time_grouped["Temperature_measured"].rolling(window).std().to_numpy()
    )
    measurements["roll_mean_voltage_60s"] = time_grouped["Voltage_measured"].rolling(window).mean().to_numpy()
    measurements["roll_std_voltage_60s"] = time_grouped["Voltage_measured"].rolling(window).std().to_numpy()
    measurements["roll_std_temperature_60s"] = measurements["roll_std_temperature_60s"].fillna(0.0)
    measurements["roll_std_voltage_60s"] = measurements["roll_std_voltage_60s"].fillna(0.0)

    measurements["time_elapsed"] = measurements["Time"] - grouped["Time"].transform("first")
    measurements["temp_rise"] = measurements["Temperature_measured"] - grouped["Temperature_measured"].transform("first")

    measurements["is_cycle_start"] = grouped.cumcount() == 0
    start_mask = measurements["is_cycle_start"]
    for column in ("dT_dt", "dV_dt", "dI_dt", "sampling_interval_s"):
        measurements.loc[start_mask, column] = 0.0
    print(f"\nis_cycle_start rows (dT_dt/dV_dt/dI_dt/sampling_interval_s set to 0.0, not dropped): {int(start_mask.sum())} of {len(measurements)}")

    return measurements


def report_b0005_rolling_window_comparison(features: pd.DataFrame, battery_id: str = "B0005") -> None:
    section(f"Time-based vs sample-based rolling window comparison ({battery_id})")
    battery_df = features[features["battery_id"] == battery_id]
    high_mode = battery_df[battery_df["sampling_interval_s"] > 15.0]
    low_mode = battery_df[(battery_df["sampling_interval_s"] <= 15.0) & (~battery_df["is_cycle_start"])]

    for label, subset in (("overall", battery_df), ("high-sampling-interval cycles (protocol change era)", high_mode), ("low-sampling-interval cycles", low_mode)):
        if subset.empty:
            continue
        for base in ("temperature", "voltage"):
            samples_col = f"roll_mean_{base}_samples"
            seconds_col = f"roll_mean_{base}_60s"
            diff = (subset[samples_col] - subset[seconds_col]).abs()
            print(f"{label} -- |{samples_col} - {seconds_col}|: mean={diff.mean():.4f}, max={diff.max():.4f} (n={len(subset)})")


def validate(features: pd.DataFrame) -> None:
    section("Validate")
    summary = features[ENGINEERED_FEATURE_COLUMNS].agg(["min", "median", "max"]).T
    summary["nulls"] = features[ENGINEERED_FEATURE_COLUMNS].isna().sum()
    print(summary.to_string())

    implausible_dT = int((features["dT_dt"].abs() > DT_DT_IMPLAUSIBLE_ABS).sum())
    voltage_lo, voltage_hi = VOLTAGE_PLAUSIBLE_RANGE
    temp_lo, temp_hi = TEMPERATURE_PLAUSIBLE_RANGE
    implausible_voltage = int((~features["Voltage_measured"].between(voltage_lo, voltage_hi)).sum())
    implausible_temp = int((~features["Temperature_measured"].between(temp_lo, temp_hi)).sum())

    print(f"\n|dT_dt| > {DT_DT_IMPLAUSIBLE_ABS} degC/s: {implausible_dT} of {len(features)} (reported, not dropped)")
    print(f"Voltage_measured outside [{voltage_lo}, {voltage_hi}] V: {implausible_voltage} of {len(features)} (reported, not dropped)")
    print(f"Temperature_measured outside [{temp_lo}, {temp_hi}] degC: {implausible_temp} of {len(features)} (reported, not dropped)")


def plot_dT_dt_distribution(features: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(features["dT_dt"].dropna(), bins=100)
    ax.set_xlabel("dT_dt (degC/s)")
    ax.set_ylabel("count")
    ax.set_title("NASA measurements: dT_dt distribution (includes is_cycle_start zeros)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "s05_dT_dt_distribution.png")
    plt.close(fig)


def plot_temperature_profile_sample(features: pd.DataFrame) -> None:
    cycle_counts = features.groupby("battery_id")["discharge_cycle_index"].nunique()
    sample_battery = cycle_counts.idxmax()
    sample = features[features["battery_id"] == sample_battery]

    fig, ax = plt.subplots(figsize=(9, 6))
    soh_values = sample.groupby("discharge_cycle_index")["soh"].first()
    norm = plt.Normalize(vmin=soh_values.min(), vmax=soh_values.max())
    cmap = plt.get_cmap("viridis")

    for cycle_index, group in sample.groupby("discharge_cycle_index"):
        group = group.sort_values("time_elapsed")
        color = cmap(norm(group["soh"].iloc[0]))
        ax.plot(group["time_elapsed"], group["Temperature_measured"], color=color, alpha=0.6, linewidth=0.8)

    ax.set_xlabel("time_elapsed (s)")
    ax.set_ylabel("Temperature_measured (degC)")
    ax.set_title(f"NASA: temperature vs time_elapsed, {sample_battery} ({len(soh_values)} cycles), coloured by soh")
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])
    fig.colorbar(mappable, ax=ax, label="soh")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "s05_temperature_profile_sample.png")
    plt.close(fig)


def plot_sampling_interval_hist(features: pd.DataFrame) -> None:
    grouped = features.groupby(["battery_id", "discharge_cycle_index"], sort=False)
    dt = grouped["Time"].diff().dropna()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(dt, bins=100)
    ax.set_xlabel("sampling interval (s)")
    ax.set_ylabel("count")
    ax.set_title("NASA measurements: sampling-interval distribution")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "s05_sampling_interval_hist.png")
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    qualifying, per_cycle_dir = load_scope()
    measurements = load_measurements(qualifying, per_cycle_dir)
    features = compute_features(measurements)
    validate(features)
    report_b0005_rolling_window_comparison(features)

    section("Write")
    output = features[
        [
            "battery_id",
            "discharge_cycle_index",
            "Time",
            "time_elapsed",
            *RAW_MEASUREMENT_COLUMNS,
            *ENGINEERED_FEATURE_COLUMNS,
            "soh",
            "ambient_temperature",
            "load_level",
            "is_cycle_start",
            "short_cycle",
        ]
    ]
    output.to_parquet(OUTPUT_PATH, index=False)
    print(f"wrote {OUTPUT_PATH}")

    section("Plots")
    plot_dT_dt_distribution(features)
    plot_temperature_profile_sample(features)
    plot_sampling_interval_hist(features)
    print(f"saved plots to {FIGURES_DIR}")

    section("FINAL SUMMARY")
    memory_mb = output.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"estimated memory footprint: {memory_mb:.1f} MB")
    print(f"final shape: {output.shape}")


if __name__ == "__main__":
    main()
