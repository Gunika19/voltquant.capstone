"""Standardise sequence features on train statistics and write the dataset card.

Reads data/04_model_input/sequences_{train,val,test}.npz, writes
data/04_model_input/scaled_{train,val,test}.npz, scaler.json, and
DATASET_CARD.md.

Per-feature mean/std are computed over (sequence, timestep) jointly on the
TRAIN split only, then applied to val/test unchanged -- fitting on the full
set (train+val+test) would leak test-split statistics (its mean, its scale)
into the transform applied to train, which is exactly the kind of leakage
assert_no_label_leakage cannot see because it's not a label column at all,
just a split-boundary violation.

No CTGAN, SMOTE, or other class-balance correction is applied here. That is
a training-time decision to make after looking at the balance this script
reports, and if applied at all it belongs inside fit_resample-on-train-only,
never touching val/test -- out of scope for this script.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from vq_core.paths import PROJECT_ROOT
from vq_core.schema import (
    CROSSING_PERSISTENCE_CYCLES,
    FRESH_STATE_MISSING,
    LABELS,
    SOH_EOL_THRESHOLD,
    WARNING_LEAD_CYCLES,
)

DATA_DIR = PROJECT_ROOT / "data" / "04_model_input"
CYCLES_PATH = PROJECT_ROOT / "data" / "02_intermediate" / "nasa_discharge_cycles.parquet"
MANIFEST_PATH = DATA_DIR / "sequence_manifest.parquet"
SCALER_PATH = DATA_DIR / "scaler.json"
DATASET_CARD_PATH = DATA_DIR / "DATASET_CARD.md"

SPLITS = ["train", "val", "test"]

# s04_build_discharge_table.py's Step 3b sensitivity check (capacity_ref
# computed 4 different ways: top-5 median, first-5 median, single max,
# nominal nameplate) found the 80% SOH crossing point shifts by more than 5
# cycles across estimators for 12 of the 34 batteries -- re-verified by
# re-running s04 in this pipeline pass; not recomputed here since that
# requires duplicating s04's four-estimator logic for a documentation
# footnote, not a modelling decision this script makes.
CAPACITY_REF_ESTIMATOR_SHIFT_CELLS = 12
CAPACITY_REF_ESTIMATOR_SHIFT_TOTAL = 34


def section(title: str) -> None:
    print()
    print(f"=== {title} ===")


def load_split(split: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    data = np.load(DATA_DIR / f"sequences_{split}.npz", allow_pickle=True)
    return data["X"], data["y"], list(data["feature_columns"])


def fit_scaler(X_train: np.ndarray, feature_columns: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    section("Fit scaler (train only)")
    flat = X_train.reshape(-1, X_train.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)

    zero_variance = [feature_columns[i] for i in np.flatnonzero(std == 0)]
    if zero_variance:
        print(f"[WARNING] zero-variance feature(s) in train: {zero_variance} -- std forced to 1.0 to avoid divide-by-zero")
        std = np.where(std == 0, 1.0, std)

    report = pd.DataFrame({"feature": feature_columns, "mean": mean, "std": std})
    print(report.to_string(index=False))
    return mean, std, zero_variance


def apply_scaler(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


def confirm_scaling(X_train_scaled: np.ndarray) -> None:
    section("Confirm scaling")
    overall_mean = X_train_scaled.mean()
    overall_std = X_train_scaled.std()
    print(f"scaled train: overall mean = {overall_mean:.6f}, overall std = {overall_std:.6f}")
    assert abs(overall_mean) < 1e-3, f"scaled train mean not ~0: {overall_mean}"
    assert abs(overall_std - 1.0) < 1e-3, f"scaled train std not ~1: {overall_std}"
    print("PASSED -- scaled train is ~mean 0, ~std 1.")


def write_scaler_json(feature_columns: list[str], mean: np.ndarray, std: np.ndarray, zero_variance: list[str]) -> None:
    payload = {
        "feature_columns": feature_columns,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "zero_variance_features": zero_variance,
        "fitted_on": "train",
    }
    SCALER_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {SCALER_PATH}")


def split_summary(manifest: pd.DataFrame) -> dict[str, dict]:
    summary = {}
    for split in SPLITS:
        split_rows = manifest[manifest["split"] == split]
        cells = sorted(split_rows["battery_id"].unique())
        counts = split_rows["label"].value_counts().sort_index()
        total = int(counts.sum())
        summary[split] = {
            "cells": cells,
            "n_windows": total,
            "counts": {LABELS[label]: int(count) for label, count in counts.items()},
            "pct": {LABELS[label]: 100 * count / total for label, count in counts.items()},
        }
    return summary


def write_dataset_card(feature_columns: list[str], summary: dict[str, dict], zero_variance: list[str]) -> None:
    cycles = pd.read_parquet(CYCLES_PATH)
    verdict_counts = cycles.groupby("battery_id")["verdict"].first().value_counts()

    lines = []
    lines.append("# NASA Li-ion battery early-warning dataset")
    lines.append("")
    lines.append("Generated by s06_build_labels.py -> s07_build_sequences.py -> s08_scale_and_finalise.py.")
    lines.append("")

    lines.append("## Label definition")
    lines.append("")
    lines.append(
        "Three-class early-warning label, assigned at discharge-cycle level then inherited by every "
        "measurement/window within that cycle:"
    )
    lines.append(f"- **{LABELS[2]}** (2): `discharge_cycle_index >= sustained_crossing_index`")
    lines.append(
        f"- **{LABELS[1]}** (1): within {WARNING_LEAD_CYCLES} cycles before the sustained crossing"
    )
    lines.append(f"- **{LABELS[0]}** (0): everything else, including all cycles of cells that never cross")
    lines.append("")
    lines.append(
        f"`sustained_crossing_index` is the first cycle where `soh < {SOH_EOL_THRESHOLD}` and it stays below "
        f"that threshold for the next {CROSSING_PERSISTENCE_CYCLES} consecutive valid cycles (not the naive "
        f"first-crossing point -- see s06_build_labels.py's naive-vs-sustained report for which cells that "
        f"moved between the two)."
    )
    lines.append("")

    lines.append("## Splits")
    lines.append("")
    lines.append("Grouped by `battery_id` (seed 42), stratified by whether the cell ever reaches runaway.")
    lines.append("")
    for split in SPLITS:
        info = summary[split]
        pct_str = ", ".join(f"{name}={pct:.1f}%" for name, pct in info["pct"].items())
        lines.append(f"- **{split}**: {len(info['cells'])} cells, {info['n_windows']} windows ({pct_str})")
        lines.append(f"  - cells: {info['cells']}")
    lines.append("")

    lines.append("## Features")
    lines.append("")
    lines.append(f"{len(feature_columns)} auto-detected numeric columns, standardised (mean 0 / std 1 on train):")
    lines.append("")
    for column in feature_columns:
        flag = " (zero-variance on train, std forced to 1.0)" if column in zero_variance else ""
        lines.append(f"- `{column}`{flag}")
    lines.append("")

    lines.append("## Known limitations")
    lines.append("")
    lines.append(
        f"- **{int(verdict_counts.get('ANOMALOUS', 0))} ANOMALOUS-verdict cells are excluded** from this "
        f"dataset entirely (s05 built the measurement table from CLEAN/STRATIFIED cells only). No held-out "
        f"edge-case set covering them has been built yet."
    )
    lines.append(
        f"- **{', '.join(FRESH_STATE_MISSING)} has no baseline** capacity reading anywhere in its recorded "
        f"history (verdict NO_BASELINE in s04) and is excluded from this dataset as a result."
    )
    lines.append(
        f"- **The {SOH_EOL_THRESHOLD:.0%} SOH crossing point is sensitive to how `capacity_ref` is estimated**: "
        f"across four candidate estimators (top-5 median, first-5 median, single max, nominal nameplate), the "
        f"crossing point shifts by more than 5 cycles for {CAPACITY_REF_ESTIMATOR_SHIFT_CELLS} of "
        f"{CAPACITY_REF_ESTIMATOR_SHIFT_TOTAL} batteries (s04_build_discharge_table.py, Step 3b). The labels "
        f"in this dataset use the top-5-median estimator throughout; a different choice would move some cells' "
        f"labels."
    )
    lines.append(
        "- **Sampling interval is a per-cycle covariate, not a constant**: it varies 9.3-20.5s across the "
        "dataset, including within a single battery's own history (see s05b_check_sampling_confound.py). "
        "`sampling_interval_s` is included as a feature so downstream models can condition on it, but any "
        "cross-cycle comparison of the raw sample-count rolling features should account for this."
    )
    lines.append(
        "- **No resampling has been applied.** Class balance is reported above as observed; CTGAN, SMOTE, or "
        "any other class-balance correction is a training-time decision to make after reviewing that balance, "
        "and if used at all must be fit on the train split only."
    )
    lines.append("")

    DATASET_CARD_PATH.write_text("\n".join(lines))
    print(f"wrote {DATASET_CARD_PATH}")


def main() -> None:
    manifest = pd.read_parquet(MANIFEST_PATH)
    summary = split_summary(manifest)

    section("Split summary (for dataset card)")
    for split in SPLITS:
        info = summary[split]
        pct_str = ", ".join(f"{name}={pct:.1f}%" for name, pct in info["pct"].items())
        print(f"{split}: {len(info['cells'])} cells, {info['n_windows']} windows ({pct_str})")

    X_train, _y_train, feature_columns = load_split("train")
    mean, std, zero_variance = fit_scaler(X_train, feature_columns)

    scaled = {}
    for split in SPLITS:
        X, y, split_features = load_split(split)
        assert split_features == feature_columns, f"{split} feature order does not match train"
        scaled[split] = (apply_scaler(X, mean, std), y)

    confirm_scaling(scaled["train"][0])

    for split in SPLITS:
        X_scaled, y = scaled[split]
        output_path = DATA_DIR / f"scaled_{split}.npz"
        np.savez(output_path, X=X_scaled, y=y, feature_columns=np.array(feature_columns))
        print(f"wrote {output_path} (X shape {X_scaled.shape})")

    write_scaler_json(feature_columns, mean, std, zero_variance)
    write_dataset_card(feature_columns, summary, zero_variance)


if __name__ == "__main__":
    main()
