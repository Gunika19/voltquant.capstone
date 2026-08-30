"""
VoltQuant — Preprocessing Pipeline Nodes
Dataset: ev_battery_degradation_v1.csv
Phase: 1 — Data Engineering
"""

import pandas as pd
import numpy as np


def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop categorical columns not useful for physical modeling."""
    cols_to_drop = ["Car_Model", "Battery_Type", "Driving_Style"]
    existing = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing)
    print(f"[drop_irrelevant_columns] Dropped: {existing}")
    print(f"[drop_irrelevant_columns] Shape: {df.shape}")
    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to unified VoltQuant standard format."""
    rename_map = {
        "Vehicle_Age_Months"     : "Age",
        "Total_Charging_Cycles"  : "Cycle_Count",
        "Avg_Temperature_C"      : "Temperature",
        "Fast_Charge_Ratio"      : "FastChargeRatio",
        "Avg_Discharge_Rate_C"   : "DischargeRate",
        "Internal_Resistance_Ohm": "Resistance",
        "SoH_Percent"            : "SOH",
    }
    df = df.rename(columns=rename_map)
    print(f"[rename_columns] Columns: {df.columns.tolist()}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Report and remove rows with missing values."""
    print(f"[handle_missing_values] Nulls before:\n{df.isnull().sum()}")
    before = len(df)
    df = df.dropna()
    after = len(df)
    print(f"[handle_missing_values] Rows dropped: {before - after} | Remaining: {after}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create health indicators, stress features, and combined stress index."""
    df["Health"]      = df["SOH"]
    df["Health_Drop"] = 1 - df["SOH"]
    df["Temperature_Stress"] = df["Temperature"] - df["Temperature"].mean()
    df["Cycle_Stress"] = df["Cycle_Count"] / (df["Age"] + 1)
    df["Stress"] = (
          0.4 * df["Temperature_Stress"]
        + 0.3 * df["Cycle_Stress"]
        + 0.3 * df["Resistance"]
    )
    print("[engineer_features] Features created: Health, Health_Drop, Temperature_Stress, Cycle_Stress, Stress")
    return df


def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Binary classification: 1 = Risk, 0 = Safe."""
    df["Label"] = (
        (df["SOH"] < 95) |
        (df["Temperature"] > 40) |
        (df["Resistance"] > df["Resistance"].mean())
    ).astype(int)
    counts = df["Label"].value_counts()
    total  = len(df)
    print(f"[create_labels] Class distribution:")
    print(f"  Safe  (0): {counts.get(0, 0)} ({counts.get(0, 0)/total*100:.1f}%)")
    print(f"  Risk  (1): {counts.get(1, 0)} ({counts.get(1, 0)/total*100:.1f}%)")
    ratio = counts.get(0, 1) / max(counts.get(1, 1), 1)
    if ratio > 3:
        print(f"  Imbalance ratio {ratio:.1f}:1 — consider SMOTE or class_weight")
    return df


def select_final_features(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the 13 final features required for modeling."""
    FINAL_FEATURES = [
        "Age", "Cycle_Count", "Temperature",
        "FastChargeRatio", "DischargeRate", "Resistance", "SOH",
        "Health", "Health_Drop",
        "Temperature_Stress", "Cycle_Stress", "Stress",
        "Label",
    ]
    available = [c for c in FINAL_FEATURES if c in df.columns]
    missing   = [c for c in FINAL_FEATURES if c not in df.columns]
    if missing:
        print(f"[select_final_features] Missing columns: {missing}")
    df = df[available]
    print(f"[select_final_features] Final shape: {df.shape}")
    print(f"[select_final_features] Columns: {df.columns.tolist()}")
    return df