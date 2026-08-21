"""Slice labelled measurements into fixed-length windows and split by cell.

Reads data/03_features/nasa_labelled_measurements.parquet, writes
data/04_model_input/sequences_{train,val,test}.npz and
data/04_model_input/sequence_manifest.parquet.

Windows are cut within a single discharge cycle and never span a cycle
boundary -- dT/dt, dV/dt, dI/dt and the rolling features all go through a
meaningless discontinuity at the gap between one discharge and the next
(different starting voltage/temperature, a rest period in between), so a
window straddling that gap would hand the model a fabricated transition.
Window length 50 samples, stride 25 (50% overlap). A window's label is
inherited from its parent cycle's label -- every measurement row in a cycle
already carries the same label (verified below), so this is just "the
cycle's label", not a new derivation.

Splitting is grouped by battery_id (a cell's cycles never span two splits)
and stratified by whether the cell ever reaches runaway, so that a cell
population this small (23 cells) can't accidentally leave a split with zero
positives.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vq_core.guards import assert_grouped_split
from vq_core.paths import PROJECT_ROOT

INPUT_PATH = PROJECT_ROOT / "data" / "03_features" / "nasa_labelled_measurements.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "04_model_input"
MANIFEST_PATH = OUTPUT_DIR / "sequence_manifest.parquet"

WINDOW_SIZE = 50
STRIDE = 25

# Non-feature identifier / bookkeeping columns to exclude even if numeric,
# because they identify a row rather than measure it. Most of these (e.g.
# record_index, verdict, uid) don't exist in nasa_labelled_measurements.parquet
# today -- they're listed anyway so this stays correct if an upstream change
# adds one without anyone having to remember to update this list too.
IDENTIFIER_COLUMNS = {
    "battery_id",
    "discharge_cycle_index",
    "record_index",
    "verdict",
    "invalid_measurement",
    "is_cycle_start",
    "short_cycle",
    "start_time",
    "uid",
    "test_id",
    "filename",
}

# Label and label-derived columns, on top of the capacity*/time_to_* prefix
# rule applied in detect_feature_columns.
LABEL_DERIVED_COLUMNS = {"label", "soh", "soh_pct", "cycles_to_crossing"}

TEST_FRACTION = 0.30
VAL_FRACTION = 0.15
SPLIT_SEED = 42


def section(title: str) -> None:
    print()
    print(f"=== {title} ===")


def detect_feature_columns(df: pd.DataFrame) -> list[str]:
    section("Feature columns")
    numeric_columns = df.select_dtypes(include="number").columns
    features = [
        column
        for column in numeric_columns
        if column not in IDENTIFIER_COLUMNS
        and column.lower() not in LABEL_DERIVED_COLUMNS
        and not column.lower().startswith("capacity")
        and not column.lower().startswith("time_to_")
    ]
    print(f"{len(features)} feature columns (auto-detected, review for leaks):")
    for column in features:
        print(f"  {column}")
    return features


def build_windows(df: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, list[np.ndarray]]:
    section("Windowing")
    manifest_rows = []
    window_arrays = []
    n_skipped_cycles = 0
    n_cycles = 0

    for (battery_id, cycle_index), group in df.groupby(["battery_id", "discharge_cycle_index"], sort=False):
        n_cycles += 1
        group = group.sort_values("Time")

        label_values = group["label"].unique()
        assert len(label_values) == 1, f"{battery_id} cycle {cycle_index} has non-uniform label {label_values}"
        label = int(label_values[0])

        values = group[feature_columns].to_numpy(dtype=np.float32)
        n_samples = len(values)
        if n_samples < WINDOW_SIZE:
            n_skipped_cycles += 1
            continue

        for start in range(0, n_samples - WINDOW_SIZE + 1, STRIDE):
            window_arrays.append(values[start : start + WINDOW_SIZE])
            manifest_rows.append(
                {
                    "battery_id": battery_id,
                    "discharge_cycle_index": cycle_index,
                    "window_start": start,
                    "window_end": start + WINDOW_SIZE,
                    "n_samples_in_cycle": n_samples,
                    "label": label,
                }
            )

    manifest = pd.DataFrame(manifest_rows)
    print(f"cycles seen: {n_cycles}")
    print(f"cycles skipped (fewer than {WINDOW_SIZE} samples): {n_skipped_cycles}")
    print(f"windows produced: {len(manifest)}")
    return manifest, window_arrays


def assign_splits(manifest: pd.DataFrame) -> dict[str, str]:
    section("Split assignment")
    cell_ever_runaway = manifest.groupby("battery_id")["label"].apply(lambda labels: bool((labels == 2).any())).to_dict()

    rng = np.random.RandomState(SPLIT_SEED)
    split_of: dict[str, str] = {}
    for stratum_flag in (True, False):
        cells = sorted(battery_id for battery_id, ever in cell_ever_runaway.items() if ever == stratum_flag)
        cells = list(cells)
        rng.shuffle(cells)
        n = len(cells)
        n_test = round(n * TEST_FRACTION)
        n_val = round(n * VAL_FRACTION)
        for battery_id in cells[:n_test]:
            split_of[battery_id] = "test"
        for battery_id in cells[n_test : n_test + n_val]:
            split_of[battery_id] = "val"
        for battery_id in cells[n_test + n_val :]:
            split_of[battery_id] = "train"
        print(
            f"stratum ever_runaway={stratum_flag}: {n} cell(s) -> "
            f"test={len(cells[:n_test])}, val={len(cells[n_test:n_test + n_val])}, "
            f"train={len(cells[n_test + n_val:])}"
        )

    return split_of


def report_splits(manifest: pd.DataFrame) -> None:
    section("Split composition")
    for split in ["train", "val", "test"]:
        cells = sorted(manifest.loc[manifest["split"] == split, "battery_id"].unique())
        counts = manifest.loc[manifest["split"] == split, "label"].value_counts().sort_index()
        total = counts.sum()
        balance = ", ".join(f"{label}={100 * count / total:.1f}%" for label, count in counts.items())
        print(f"{split}: {len(cells)} cell(s) -> {cells}")
        print(f"  {total} windows, class balance: {balance}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(INPUT_PATH)
    feature_columns = detect_feature_columns(df)

    manifest, window_arrays = build_windows(df, feature_columns)

    split_of = assign_splits(manifest)
    manifest["split"] = manifest["battery_id"].map(split_of)
    assert_grouped_split(manifest, split_column="split", group_column="battery_id")
    print("\nassert_grouped_split: PASSED -- no cell appears in more than one split.")

    report_splits(manifest)

    manifest["sequence_index"] = -1
    for split in ["train", "val", "test"]:
        split_mask = manifest["split"] == split
        positions = np.flatnonzero(split_mask.to_numpy())
        X = np.stack([window_arrays[position] for position in positions]).astype(np.float32)
        y = manifest.loc[split_mask, "label"].to_numpy(dtype=np.int64)
        manifest.loc[split_mask, "sequence_index"] = np.arange(len(positions))

        output_path = OUTPUT_DIR / f"sequences_{split}.npz"
        np.savez(output_path, X=X, y=y, feature_columns=np.array(feature_columns))
        print(f"\nwrote {output_path} (X shape {X.shape}, y shape {y.shape})")

    manifest.to_parquet(MANIFEST_PATH, index=False)
    print(f"wrote {MANIFEST_PATH} ({len(manifest)} rows)")

    section("Note")
    print(
        "The 8 ANOMALOUS-verdict cells are absent from this dataset entirely -- s05 built "
        "nasa_discharge_measurements.parquet from CLEAN/STRATIFIED cells only, so there is no "
        "measurement data for them to sequence here. A held-out edge-case set covering those "
        "cells has not been built yet."
    )


if __name__ == "__main__":
    main()
