"""Verify the raw NASA download is ready for the pipeline.

Locates metadata.csv and the per-cycle data directory, inventories the raw
files, and runs four gates: chronology, monotonicity, sparsity, and sample
budget. Prints a report to stdout and writes plots to reports/figures/.
Read-only -- writes no data files.
"""

from __future__ import annotations

import itertools
import random
import re
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm
from vq_core.paths import PROJECT_ROOT, RAW_NASA

FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# start_time is a stringified 6-element array (year, month, day, hour, minute,
# second) but numpy's repr varies -- plain, zero-padded, or scientific
# notation -- so pull out numeric tokens instead of assuming one layout.
START_TIME_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

METADATA_SEARCH_DEPTH = 3
SAMPLE_FILE_COUNT = 30
SAMPLE_SEED = 0
SEQUENCE_WINDOW = 10
SEQUENCE_STRIDE = 1

verdict_gate: str | None = None


def section(title: str) -> None:
    print()
    print(f"=== {title} ===")


def block(gate: str, reason: str) -> None:
    global verdict_gate
    verdict_gate = gate
    print()
    print(f"*** BLOCKED at {gate}: {reason} ***")


def find_metadata_csv(root: Path, max_depth: int) -> Path | None:
    for depth in range(max_depth + 1):
        pattern = "/".join(["*"] * depth + ["metadata.csv"])
        matches = sorted(root.glob(pattern))
        if matches:
            return matches[0]
    return None


def find_per_cycle_dir(search_root: Path, sample_filename: str) -> Path | None:
    matches = list(search_root.rglob(sample_filename))
    return matches[0].parent if matches else None


def parse_start_time(raw: str) -> datetime:
    numbers = [float(token) for token in START_TIME_NUMBER_RE.findall(raw)]
    if len(numbers) < 6:
        raise ValueError(f"expected 6 numbers in start_time, got {numbers} from {raw!r}")
    year, month, day, hour, minute, second = numbers[:6]
    # int(second) below truncates any fractional seconds -- recover them here
    # first, or they'd be silently lost rather than rounded into microsecond.
    microsecond = round((second % 1) * 1_000_000)
    # start_time is a naive local timestamp -- NASA's extra_infos READMEs and
    # metadata.csv record no timezone anywhere in this dataset, so attaching
    # one (e.g. UTC) would assert precision the source doesn't have. Used
    # only for within-battery chronological ordering, never absolute time
    # arithmetic, so naive is correct here, not an oversight.
    return datetime(int(year), int(month), int(day), int(hour), int(minute), int(second), microsecond)  # noqa: DTZ001


def is_present(series: pd.Series) -> pd.Series:
    """Non-missing, treating the literal string '[]' as missing like NaN."""
    return series.notna() & (series.astype(str).str.strip() != "[]")


def discover() -> tuple[Path | None, Path | None]:
    section("Discovery")
    metadata_path = find_metadata_csv(RAW_NASA, METADATA_SEARCH_DEPTH)
    if metadata_path is None:
        print(f"metadata.csv not found within {METADATA_SEARCH_DEPTH} levels of {RAW_NASA}")
        return None, None

    print(f"metadata.csv found at: {metadata_path}")
    actual_dir = metadata_path.parent
    if actual_dir != RAW_NASA:
        print(
            f"RAW_NASA in vq_core.paths ({RAW_NASA}) does NOT point at the directory "
            f"containing metadata.csv. Use {actual_dir} instead -- update RAW_NASA, "
            f"don't work around it here."
        )
    else:
        print(f"RAW_NASA correctly points at the directory containing metadata.csv ({RAW_NASA}).")

    mat_files = list(RAW_NASA.rglob("*.mat"))
    print(f".mat files present under RAW_NASA: {len(mat_files)}")

    return metadata_path, actual_dir


def inventory(metadata_path: Path, metadata_dir: Path) -> tuple[pd.DataFrame, Path | None]:
    section("Inventory")
    df = pd.read_csv(metadata_path)
    print(f"shape: {df.shape}")
    print(f"columns ({len(df.columns)}): {list(df.columns)}")

    print("\nrows per battery_id:")
    print(df.groupby("battery_id").size())

    print("\ntype value counts:")
    print(df["type"].value_counts())

    print("\nambient_temperature value counts:")
    print(df["ambient_temperature"].value_counts().sort_index())

    referenced = set(df["filename"].dropna())
    sample_filename = next(iter(referenced))
    per_cycle_dir = find_per_cycle_dir(metadata_dir, sample_filename)
    if per_cycle_dir is None:
        print(f"\ncould not locate the per-cycle data directory (looked for {sample_filename})")
        return df, None

    on_disk = {p.name for p in per_cycle_dir.glob("*.csv")}
    print(f"\nper-cycle data directory: {per_cycle_dir}")
    print(f"files on disk: {len(on_disk)}")
    print(f"referenced filenames in metadata: {len(referenced)}")
    print(f"referenced & present: {len(referenced & on_disk)}")
    print(f"referenced but missing from disk: {len(referenced - on_disk)}")
    print(f"present on disk but unreferenced: {len(on_disk - referenced)}")

    return df, per_cycle_dir


def gate_a_chronology(df: pd.DataFrame) -> pd.DataFrame | None:
    section("Gate A: chronology")
    if "start_time" not in df.columns:
        print("start_time column is ABSENT from metadata.")
        block("Gate A (chronology)", "no start_time column -- cannot order cycles")
        return None

    print("start_time column present. Parsing and deriving cycle_index per battery.")
    df = df.copy()
    df["_start_dt"] = df["start_time"].map(parse_start_time)
    df["cycle_index"] = (
        df.groupby("battery_id")["_start_dt"].rank(method="first").astype(int) - 1
    )
    print("parsed start_time for all rows; cycle_index = rank of start_time within battery_id.")
    return df


def gate_b_monotonicity(df: pd.DataFrame) -> pd.Series | None:
    section("Gate B: monotonicity (discharge rows only)")
    df["Capacity_num"] = pd.to_numeric(df["Capacity"], errors="coerce")
    discharge = df[df["type"] == "discharge"].dropna(subset=["Capacity_num"])

    if discharge.empty:
        block("Gate B (monotonicity)", "no discharge rows with a numeric Capacity value")
        return None

    correlations = {}
    for battery_id, group in discharge.groupby("battery_id"):
        if group["cycle_index"].nunique() > 1:
            correlations[battery_id] = group["cycle_index"].corr(group["Capacity_num"], method="spearman")
    corr_series = pd.Series(correlations, name="spearman_r").sort_values()

    print("per-battery Spearman correlation (cycle_index vs Capacity):")
    print(corr_series)
    print("\ndistribution:")
    print(corr_series.describe())

    below = int((corr_series < -0.8).sum())
    mid = int(((corr_series >= -0.8) & (corr_series <= -0.3)).sum())
    above = int((corr_series > -0.3).sum())
    print(f"\nbelow -0.8 (strong degradation trend): {below}")
    print(f"between -0.8 and -0.3: {mid}")
    print(f"above -0.3 (weak/no trend): {above}")

    plot_capacity_vs_cycle(discharge)
    return corr_series


def plot_capacity_vs_cycle(discharge: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    temps = sorted(discharge["ambient_temperature"].unique())
    cmap = plt.get_cmap("viridis", len(temps))
    boundaries = [temps[0] - 0.5] + [(a + b) / 2 for a, b in itertools.pairwise(temps)] + [temps[-1] + 0.5]
    norm = BoundaryNorm(boundaries=boundaries, ncolors=len(temps))

    for _, group in discharge.groupby("battery_id"):
        group = group.sort_values("cycle_index")
        x = group["cycle_index"].to_numpy()
        y = group["Capacity_num"].to_numpy()
        temp = group["ambient_temperature"].to_numpy()
        if len(x) < 2:
            continue
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(temp[:-1])
        lc.set_linewidth(1.2)
        ax.add_collection(lc)

    ax.set_xlim(discharge["cycle_index"].min(), discharge["cycle_index"].max())
    ax.set_ylim(discharge["Capacity_num"].min() * 0.95, discharge["Capacity_num"].max() * 1.05)
    ax.set_xlabel("cycle_index")
    ax.set_ylabel("Capacity")
    ax.set_title("NASA: Capacity vs cycle_index by battery, coloured by ambient_temperature")

    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, ticks=temps)
    cbar.set_label("ambient_temperature")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "s02_capacity_vs_cycle_index.png")
    plt.close(fig)


def gate_c_sparsity(df: pd.DataFrame) -> None:
    section("Gate C: sparsity (no forward-fill)")
    for column in ["Capacity", "Re", "Rct"]:
        present = is_present(df[column])
        print(f"\n{column} presence by type:")
        print(pd.crosstab(df["type"], present))


def gate_d_sample_budget(df: pd.DataFrame, per_cycle_dir: Path) -> None:
    section("Gate D: sample budget")

    filenames = df["filename"].dropna().tolist()
    rng = random.Random(SAMPLE_SEED)
    sample_filenames = rng.sample(filenames, min(SAMPLE_FILE_COUNT, len(filenames)))
    type_by_filename = df.set_index("filename")["type"]

    columns_by_type: dict[str, set[tuple[str, ...]]] = {}
    rows_by_type: dict[str, list[int]] = {}
    for filename in sample_filenames:
        file_type = type_by_filename.get(filename, "unknown")
        sample_df = pd.read_csv(per_cycle_dir / filename)
        columns_by_type.setdefault(file_type, set()).add(tuple(sample_df.columns))
        rows_by_type.setdefault(file_type, []).append(len(sample_df))

    print(f"sampled {len(sample_filenames)} of {len(filenames)} per-cycle files (seed={SAMPLE_SEED})")
    for file_type in sorted(columns_by_type):
        variants = columns_by_type[file_type]
        rows = pd.Series(rows_by_type[file_type])
        print(f"\n{file_type}: {len(variants)} distinct column-set variant(s) in sample")
        for variant in variants:
            print(f"  columns: {list(variant)}")
        print(f"  rows per file -- min: {rows.min()}, median: {rows.median()}, max: {rows.max()}")

    mean_rows_by_type = {ft: pd.Series(rows).mean() for ft, rows in rows_by_type.items()}
    type_counts = df["type"].value_counts()
    estimated_total_rows = sum(
        mean_rows_by_type.get(file_type, 0) * count for file_type, count in type_counts.items()
    )

    discharge_counts_per_battery = df[df["type"] == "discharge"].groupby("battery_id").size()
    sequence_counts_per_battery = ((discharge_counts_per_battery - SEQUENCE_WINDOW) // SEQUENCE_STRIDE + 1).clip(
        lower=0
    )
    n_batteries = df["battery_id"].nunique()

    section("Gate D: sample counts by granularity")
    print(f"measurement level (estimated, from {SAMPLE_FILE_COUNT}-file sample): ~{estimated_total_rows:,.0f} rows")
    print(f"cycle level (discharge cycles only): {int((df['type'] == 'discharge').sum())} cycles")
    print(
        f"sequence level (window={SEQUENCE_WINDOW}, stride={SEQUENCE_STRIDE}, "
        f"discharge cycles only): {int(sequence_counts_per_battery.sum())} sequences"
    )
    print(f"\n*** independent batteries: {n_batteries} -- this governs the grouped split ***")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    metadata_path, metadata_dir = discover()
    if metadata_path is None:
        block("Discovery", "metadata.csv not found")
        print_verdict()
        return

    df, per_cycle_dir = inventory(metadata_path, metadata_dir)

    df = gate_a_chronology(df)
    if df is None:
        print_verdict()
        return

    gate_b_monotonicity(df)
    if verdict_gate is not None:
        print_verdict()
        return

    gate_c_sparsity(df)

    if per_cycle_dir is not None:
        gate_d_sample_budget(df, per_cycle_dir)
    else:
        block("Gate D (sample budget)", "per-cycle data directory not found")

    print_verdict()


def print_verdict() -> None:
    section("RESULT")
    if verdict_gate is not None:
        print(f"BLOCKED -- failed at {verdict_gate}")
        raise SystemExit(1)
    print("PROCEED -- all gates passed")


if __name__ == "__main__":
    main()
