"""Cross-cutting safety checks for tables about to be fed to a model."""

import warnings

import pandas as pd

from vq_core.schema import DROP_ON_LOAD, LABEL_ONLY_FEATURES


class LabelLeakageError(Exception):
    """Raised when a table about to be used as model input contains a label-only column."""


def assert_source_columns_dropped(df: pd.DataFrame, source_key: str) -> None:
    """Raise if any DROP_ON_LOAD column for `source_key` is still present in `df`.

    This only knows about the columns listed in DROP_ON_LOAD (the synthetic EV
    sources' direct leaks and non-measurement setpoints) -- it says nothing
    about capacity/re/rct. Use assert_no_label_leakage for that.
    """
    forbidden = DROP_ON_LOAD.get(source_key, [])
    present = [column for column in forbidden if column in df.columns]
    if present:
        raise AssertionError(f"leakage: {source_key} table still contains DROP_ON_LOAD columns {present}")


def assert_no_label_leakage(df: pd.DataFrame) -> None:
    """Raise LabelLeakageError if `df` contains any LABEL_ONLY_FEATURES column.

    Matches case-insensitively, so both the raw NASA casing (Capacity, Re, Rct)
    and the lowercase columns in nasa_discharge_cycles.parquet (capacity, soh,
    soh_pct, ...) are caught regardless of which table this is called on.
    """
    forbidden_lower = {name.lower() for name in LABEL_ONLY_FEATURES}
    offending = [column for column in df.columns if column.lower() in forbidden_lower]
    if offending:
        raise LabelLeakageError(f"label leakage: model input contains label-only column(s) {offending}")


def assert_model_input_ready(df: pd.DataFrame, feature_columns: list[str]) -> None:
    """Raise if `df` leaks a label column or is missing a declared feature column.

    Warns (does not raise) if `df` has columns present but not declared in
    `feature_columns` -- those are silently ignored by anything that selects
    only feature_columns, so they're a smell worth surfacing, not a hard stop.
    """
    assert_no_label_leakage(df)

    missing = [column for column in feature_columns if column not in df.columns]
    if missing:
        raise AssertionError(f"model input missing declared feature column(s) {missing}")

    undeclared = [column for column in df.columns if column not in feature_columns]
    if undeclared:
        warnings.warn(f"model input has column(s) present but not declared as a feature: {undeclared}", stacklevel=2)


def assert_grouped_split(df: pd.DataFrame, split_column: str, group_column: str) -> None:
    """Raise if any `group_column` value appears under more than one `split_column` value."""
    groups_per_split = df.groupby(group_column)[split_column].nunique()
    leaking = groups_per_split[groups_per_split > 1]
    if not leaking.empty:
        raise AssertionError(
            f"grouped-split violation: {group_column} value(s) {list(leaking.index)} "
            f"appear under more than one {split_column}"
        )
