"""Unit tests for the VoltQuant unified evaluation module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from vq_core.evaluation import (
    aggregate_battery_predictions,
    calculate_battery_metrics,
    calculate_early_warning_metrics,
    calculate_window_metrics,
)


def test_calculate_window_metrics():
    """Verify window-level metric computation correctness."""
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    
    metrics = calculate_window_metrics(y_true, y_pred)
    
    assert "accuracy" in metrics
    assert "balanced_accuracy" in metrics
    assert "macro_f1" in metrics
    assert "per_class" in metrics
    assert "confusion_matrix" in metrics
    
    # 4 correct out of 6 -> accuracy = 2/3
    assert abs(metrics["accuracy"] - 4/6) < 1e-5
    assert len(metrics["confusion_matrix"]) == 3


def test_battery_level_aggregation_and_metrics():
    """Verify battery-level averaging and classification metrics."""
    # Create sample predictions DataFrame
    data = {
        "battery_id": ["B0001", "B0001", "B0002", "B0002"],
        "true_label": [2, 2, 0, 0],
        "predicted_label": [2, 1, 0, 0],
        "probability_class_0": [0.1, 0.2, 0.8, 0.9],
        "probability_class_1": [0.2, 0.5, 0.1, 0.05],
        "probability_class_2": [0.7, 0.3, 0.1, 0.05],
    }
    df = pd.DataFrame(data)
    
    # Aggregate predictions
    agg = aggregate_battery_predictions(df)
    
    # B0001 average probabilities: class 2 is (0.7 + 0.3)/2 = 0.5, class 1 is (0.2 + 0.5)/2 = 0.35
    # So B0001 should aggregate to class 2
    assert agg.loc[agg["battery_id"] == "B0001", "predicted_label"].values[0] == 2
    # B0002 average probabilities: class 0 is (0.8 + 0.9)/2 = 0.85
    assert agg.loc[agg["battery_id"] == "B0002", "predicted_label"].values[0] == 0
    
    # Calculate battery metrics
    metrics = calculate_battery_metrics(agg)
    assert metrics["battery_accuracy"] == 1.0
    assert metrics["critical_class_recall"] == 1.0


def test_early_warning_metrics():
    """Verify early warning prognosis metrics and warning persistence."""
    data = {
        "battery_id": ["B0001"] * 10 + ["B0002"] * 10,
        "discharge_cycle_index": list(range(10)) + list(range(10)),
        # B0001 crossings at cycle index 6
        # B0002 never crosses (all normal)
        "true_label": [0, 0, 0, 1, 1, 1, 2, 2, 2, 2] + [0]*10,
        # B0001 first predicted warning index is cycle 4 (persistent for 2 cycles)
        "predicted_label": [0, 0, 0, 0, 1, 1, 2, 2, 2, 2] + [0]*10,
    }
    df = pd.DataFrame(data)
    
    ew = calculate_early_warning_metrics(df, warning_persistence_cycles=2)
    
    # B0001 crossing at 6, warning at 4 -> lead time = 2 cycles, detected = True
    assert ew["per_battery"]["B0001"]["true_crossing_cycle"] == 6
    assert ew["per_battery"]["B0001"]["first_persistent_warning"] == 4
    assert ew["per_battery"]["B0001"]["lead_time_cycles"] == 2
    assert ew["per_battery"]["B0001"]["detected"] is True
    assert ew["per_battery"]["B0001"]["false_alarm"] is False
    
    # B0002 normal, no warning triggers -> detected = False, false_alarm = False
    assert ew["per_battery"]["B0002"]["true_crossing_cycle"] is None
    assert ew["per_battery"]["B0002"]["first_persistent_warning"] is None
    assert ew["per_battery"]["B0002"]["detected"] is False
    assert ew["per_battery"]["B0002"]["false_alarm"] is False
    
    # Overall summary metrics
    assert ew["summary"]["detection_rate"] == 1.0
    assert ew["summary"]["false_alarm_rate"] == 0.0
    assert ew["summary"]["mean_lead_time"] == 2.0


def test_battery_disjointness_helper():
    """Verify assertion functions identify overlapping batteries across splits."""
    from vq_core.guards import assert_grouped_split
    
    df_good = pd.DataFrame({
        "battery_id": ["B05", "B05", "B06", "B07"],
        "split": ["train", "train", "val", "test"]
    })
    # Should pass without raising
    assert_grouped_split(df_good, "split", "battery_id")
    
    df_bad = pd.DataFrame({
        "battery_id": ["B05", "B05", "B06", "B05"],
        "split": ["train", "train", "val", "test"]
    })
    # Should raise AssertionError due to B05 in both train and test splits
    with pytest.raises(AssertionError):
        assert_grouped_split(df_bad, "split", "battery_id")
