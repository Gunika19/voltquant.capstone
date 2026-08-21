import pandas as pd
import pytest
from vq_core.guards import (
    LabelLeakageError,
    assert_grouped_split,
    assert_no_label_leakage,
)


@pytest.mark.parametrize("column_name", ["capacity", "Capacity"])
def test_capacity_column_raises_regardless_of_case(column_name):
    df = pd.DataFrame({"voltage": [3.7, 3.6], column_name: [1.5, 1.4]})
    with pytest.raises(LabelLeakageError):
        assert_no_label_leakage(df)


def test_soh_pct_column_raises():
    df = pd.DataFrame({"voltage": [3.7, 3.6], "soh_pct": [95.0, 94.0]})
    with pytest.raises(LabelLeakageError):
        assert_no_label_leakage(df)


def test_time_to_cutoff_column_raises():
    df = pd.DataFrame({"voltage": [3.7, 3.6], "time_to_cutoff_s": [1500.0, 1200.0]})
    with pytest.raises(LabelLeakageError):
        assert_no_label_leakage(df)


def test_clean_measurement_frame_passes():
    df = pd.DataFrame(
        {
            "voltage": [3.7, 3.6, 3.5],
            "current": [-2.0, -2.0, -1.9],
            "temperature": [24.1, 24.3, 24.5],
        }
    )
    assert_no_label_leakage(df)  # must not raise


def test_grouped_split_catches_battery_in_both_train_and_test():
    df = pd.DataFrame(
        {
            "battery_id": ["B0001", "B0001", "B0002", "B0002"],
            "split": ["train", "test", "train", "train"],
        }
    )
    with pytest.raises(AssertionError):
        assert_grouped_split(df, split_column="split", group_column="battery_id")


def test_grouped_split_passes_when_groups_are_disjoint():
    df = pd.DataFrame(
        {
            "battery_id": ["B0001", "B0001", "B0002", "B0002"],
            "split": ["train", "train", "test", "test"],
        }
    )
    assert_grouped_split(df, split_column="split", group_column="battery_id")  # must not raise
