"""Trial 3: Data Ingestion & Imputation Pipeline (MIT LFP Fast-Charging Cohort)

Loads the clean and raw 20-feature dataset (Severson et al., Nature Energy 2019),
verifies column schemas and cohort distributions, and applies causal median imputation
for the 2 missing resistance readings on cell b1c18 (ir_cycle2 and ir_diff_100_2).

Outputs:
  - trial_03_mit_lfp/data/mit_cycle_life_processed.parquet
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "data"

CLEAN_CSV_PATH = DATA_DIR / "mit_cycle_life_clean.csv"
RAW_CSV_PATH = DATA_DIR / "mit_cycle_life_raw.csv"
SPLITS_CSV_PATH = DATA_DIR / "mit_splits_70_30_x20.csv"
OUTPUT_PARQUET_PATH = DATA_DIR / "mit_cycle_life_processed.parquet"

FEATURE_COLS = [
    "dq_min", "dq_mean", "dq_var", "dq_skew", "dq_kurt", "dq_2v",
    "cap_slope_2_100", "cap_intercept_2_100", "cap_slope_91_100", "cap_intercept_91_100",
    "q_cycle2", "qmax_minus_q2", "q_cycle100", "charge_time_avg",
    "temp_max", "temp_min", "temp_integral",
    "ir_cycle2", "ir_min", "ir_diff_100_2"
]

SEVERSON_MODELS = {
    "variance": ["dq_var"],
    "discharge": ["dq_min", "dq_var", "dq_skew", "dq_kurt", "q_cycle2", "qmax_minus_q2"],
    "full": [
        "dq_min", "dq_var", "cap_slope_2_100", "cap_intercept_2_100",
        "q_cycle2", "charge_time_avg", "temp_integral", "ir_min", "ir_diff_100_2"
    ]
}


def ingest_and_impute() -> pd.DataFrame:
    print("=" * 70)
    print("M01 · MIT LFP DATA INGESTION & QUALITY IMPUTATION")
    print("=" * 70)

    if not CLEAN_CSV_PATH.exists():
        raise FileNotFoundError(f"Missing clean CSV at {CLEAN_CSV_PATH}")

    df_clean = pd.read_csv(CLEAN_CSV_PATH)
    print(f"Loaded clean dataset: {df_clean.shape[0]} cells, {df_clean.shape[1]} columns")

    # Verify cohort distribution
    split_counts = df_clean["severson_split"].value_counts().to_dict()
    print(f"Cohort splits: {split_counts}")
    assert split_counts.get("train") == 41, f"Expected 41 train cells, got {split_counts.get('train')}"
    assert split_counts.get("primary_test") == 43, f"Expected 43 primary test cells, got {split_counts.get('primary_test')}"
    assert split_counts.get("secondary_test") == 40, f"Expected 40 secondary test cells, got {split_counts.get('secondary_test')}"

    # Missing value audit
    nulls = df_clean[FEATURE_COLS].isnull().sum()
    null_cols = nulls[nulls > 0].to_dict()
    print(f"Missing values before imputation: {null_cols}")

    # Impute missing values for cell b1c18 using Batch 1 medians (training distribution)
    # This prevents NaN errors while maintaining causal data boundaries
    b1_mask = df_clean["batch"] == "b1"
    for col in ["ir_cycle2", "ir_diff_100_2"]:
        if col in null_cols:
            b1_median = df_clean.loc[b1_mask & df_clean[col].notnull(), col].median()
            impute_count = df_clean[col].isnull().sum()
            df_clean[col] = df_clean[col].fillna(b1_median)
            print(f"  -> Imputed {impute_count} missing value(s) in '{col}' using Batch 1 median ({b1_median:.6f})")

    # Verify zero missing values remaining
    remaining_nulls = df_clean[FEATURE_COLS].isnull().sum().sum()
    assert remaining_nulls == 0, f"Expected 0 remaining nulls, found {remaining_nulls}"
    print("Data quality check passed: 0 missing values across all 20 features.")

    # Save processed parquet
    df_clean.to_parquet(OUTPUT_PARQUET_PATH, index=False)
    print(f"Saved processed dataset to: {OUTPUT_PARQUET_PATH}")
    return df_clean


if __name__ == "__main__":
    ingest_and_impute()

