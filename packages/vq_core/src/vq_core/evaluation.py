"""Unified evaluation and benchmarking suite for VoltQuant prognostic models.

Provides standard calculators for:
1. Window-level metrics (balanced accuracy, macro/weighted F1, confusion matrices).
2. Battery-level metrics (aggregating overlapping sequences).
3. Early-warning metrics (lead times, false alarm rates, detection rates).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from vq_core.config import (
    WARNING_LEAD_CYCLES,
)


def calculate_window_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute standard classification metrics across all sequence windows."""
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    # Compute macro and weighted averages
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    _, _, weighted_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    
    # Compute per-class values
    per_class_p, per_class_r, per_class_f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )
    
    conf = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    
    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": {
            "precision": [float(v) for v in per_class_p],
            "recall": [float(v) for v in per_class_r],
            "f1": [float(v) for v in per_class_f1],
            "support": [int(v) for v in support],
        },
        "confusion_matrix": conf.tolist(),
    }


def aggregate_battery_predictions(df_preds: pd.DataFrame) -> pd.DataFrame:
    """Aggregate window-level predictions to battery-level using maximum state reached.
    
    Expects df_preds to contain:
        battery_id, true_label, predicted_label,
        probability_class_0, probability_class_1, probability_class_2
    
    Returns DataFrame indexed by battery_id containing:
        true_label: 2 if the battery ever reaches EOL degradation proxy, else 1 if ever warning, else 0
        predicted_label: 2 if the model ever predicts EOL degradation proxy, else 1 if ever warning, else 0
    """
    agg = df_preds.groupby("battery_id").agg(
        avg_p0=("probability_class_0", "mean"),
        avg_p1=("probability_class_1", "mean"),
        avg_p2=("probability_class_2", "mean"),
        true_label=("true_label", lambda x: 2 if 2 in x.values else (1 if 1 in x.values else 0)),
        predicted_label=("predicted_label", lambda x: 2 if 2 in x.values else (1 if 1 in x.values else 0))
    )
    return agg.reset_index()


def calculate_battery_metrics(agg_df: pd.DataFrame) -> dict:
    """Compute standard classification metrics over battery-level aggregated states."""
    y_true = agg_df["true_label"].to_numpy()
    y_pred = agg_df["predicted_label"].to_numpy()
    
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    _, _, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    
    # Focus on the critical Class 2 (runaway) performance
    p_class, r_class, f_class, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    
    return {
        "battery_accuracy": float(accuracy),
        "battery_balanced_accuracy": float(balanced_acc),
        "battery_macro_f1": float(macro_f1),
        "critical_class_precision": float(p_class[2]),
        "critical_class_recall": float(r_class[2]),
        "critical_class_f1": float(f_class[2]),
    }


def calculate_early_warning_metrics(
    df_preds: pd.DataFrame,
    warning_persistence_cycles: int = 2
) -> dict:
    """Compute early warning metrics from a DataFrame of cycle-level predictions.
    
    Requires columns:
        battery_id, discharge_cycle_index, true_label, predicted_label
    
    For each test battery:
        - true crossing cycle: first cycle where true_label is 2.
        - first persistent warning: first cycle index where predicted_label >= 1
          and remains >= 1 for the next `warning_persistence_cycles` consecutive cycles.
        - lead time: true crossing cycle - first persistent warning cycle.
        - detected: True if first persistent warning occurs before or at true crossing cycle.
        - false alarm: True if warning triggers when the cell never reaches runaway (true_label never becomes 2),
                       or if the warning is triggered way too early (before warning lead cycle window).
    """
    results = {}
    
    for battery_id, group in df_preds.groupby("battery_id"):
        # Sort chronologically by discharge cycle index
        group = group.sort_values("discharge_cycle_index")
        cycles = group["discharge_cycle_index"].to_numpy()
        trues = group["true_label"].to_numpy()
        preds = group["predicted_label"].to_numpy()
        
        # True crossing cycle index (first Class 2 label)
        cross_idx = np.where(trues == 2)[0]
        true_crossing_cycle = int(cycles[cross_idx[0]]) if len(cross_idx) > 0 else None
        
        # Persistent warning cycle index (predicted warning/runaway state >= 1)
        warning_cycle = None
        for i in range(len(preds) - warning_persistence_cycles + 1):
            if all(preds[i + k] >= 1 for k in range(warning_persistence_cycles)):
                warning_cycle = int(cycles[i])
                break
        
        lead_time = None
        detected = False
        false_alarm = False
        
        if true_crossing_cycle is not None:
            if warning_cycle is not None:
                if warning_cycle <= true_crossing_cycle:
                    lead_time = true_crossing_cycle - warning_cycle
                    detected = True
                else:
                    # Warning triggered after EOL has already crossed
                    detected = False
                
                # False alarm if triggered before warning lead cycle window starts
                warning_horizon_start = true_crossing_cycle - WARNING_LEAD_CYCLES
                if warning_cycle < warning_horizon_start:
                    false_alarm = True
            else:
                detected = False
        else:
            # Cell never reaches EOL runaway in ground truth
            if warning_cycle is not None:
                false_alarm = True
                
        results[battery_id] = {
            "true_crossing_cycle": true_crossing_cycle,
            "first_persistent_warning": warning_cycle,
            "lead_time_cycles": lead_time,
            "detected": detected,
            "false_alarm": false_alarm
        }
        
    # Aggregate early-warning stats across all test batteries
    df_warning = pd.DataFrame.from_dict(results, orient="index")
    
    valid_lead_times = df_warning["lead_time_cycles"].dropna().to_numpy()
    
    n_crossings = (df_warning["true_crossing_cycle"].notna()).sum()
    n_detections = df_warning["detected"].sum()
    detection_rate = float(n_detections / n_crossings) if n_crossings > 0 else 0.0
    
    n_false_alarms = df_warning["false_alarm"].sum()
    
    # False alarm rate: percentage of non-crossing cells that false-alarmed plus premature warnings on crossing cells
    total_cells = len(df_warning)
    false_alarm_rate = float(n_false_alarms / total_cells) if total_cells > 0 else 0.0
    
    mean_lead = float(np.mean(valid_lead_times)) if len(valid_lead_times) > 0 else None
    median_lead = float(np.median(valid_lead_times)) if len(valid_lead_times) > 0 else None
    
    return {
        "per_battery": results,
        "summary": {
            "detection_rate": detection_rate,
            "mean_lead_time": mean_lead,
            "median_lead_time": median_lead,
            "false_alarm_rate": false_alarm_rate,
        }
    }


def save_evaluation_results(
    df_preds: pd.DataFrame,
    output_dir: Path | str,
    model_name: str,
    seed: int
) -> None:
    """Save raw predictions and computed metrics to the final experiment results directory."""
    output_dir = Path(output_dir)
    pred_dir = output_dir / "predictions"
    metric_dir = output_dir / "metrics"
    
    pred_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)
    
    # Save predictions.csv
    pred_path = pred_dir / f"{model_name}_seed{seed}_predictions.csv"
    df_preds.to_csv(pred_path, index=False)
    print(f"Saved raw predictions to {pred_path}")
    
    # Calculate window, battery, and warning metrics
    window_metrics = calculate_window_metrics(df_preds["true_label"].to_numpy(), df_preds["predicted_label"].to_numpy())
    agg_df = aggregate_battery_predictions(df_preds)
    battery_metrics = calculate_battery_metrics(agg_df)
    warning_metrics = calculate_early_warning_metrics(df_preds)
    
    summary = {
        "model": model_name,
        "seed": seed,
        "window_metrics": window_metrics,
        "battery_metrics": battery_metrics,
        "early_warning_metrics": {
            "per_battery": warning_metrics["per_battery"],
            "summary": warning_metrics["summary"]
        }
    }
    
    metric_path = metric_dir / f"{model_name}_seed{seed}_metrics.json"
    with open(metric_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved metrics summary to {metric_path}")
