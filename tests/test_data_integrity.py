"""VoltQuant Test Suite: Data Integrity & Imputation Verification
Tests:
  1. Processed Parquet exists and has exactly 124 cells.
  2. Zero NaNs across all 26 feature columns.
  3. b1c18 median imputation satisfies Batch 1 training isolation (zero test leakage).
  4. Pearson correlation between log10(dq_var) and log10(cycle_life) is strongly negative (r < -0.90).
"""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PARQUET = BASE_DIR / "trial_03_mit_lfp" / "data" / "mit_cycle_life_processed.parquet"

class TestDataIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert DATA_PARQUET.exists(), f"Missing Parquet: {DATA_PARQUET}"
        cls.df = pd.read_parquet(DATA_PARQUET)

    def test_cell_count(self):
        """Check total cell count is exactly 124."""
        self.assertEqual(len(self.df), 124, f"Expected 124 cells, found {len(self.df)}")

    def test_zero_nans(self):
        """Verify zero NaN values exist in the processed dataset."""
        nan_counts = self.df.isna().sum().sum()
        self.assertEqual(nan_counts, 0, f"Found {nan_counts} NaNs in processed dataset")

    def test_b1c18_imputation(self):
        """Verify b1c18 imputation matches Batch 1 training median."""
        b1c18_row = self.df[self.df["cell_id"] == "b1c18"].iloc[0]
        # Batch 1 median for ir_cycle2 was ~0.016721
        self.assertAlmostEqual(b1c18_row["ir_cycle2"], 0.016721, places=4,
                               msg="b1c18 ir_cycle2 imputation does not match Batch 1 median")
        self.assertAlmostEqual(b1c18_row["ir_diff_100_2"], -0.000003, places=5,
                               msg="b1c18 ir_diff_100_2 imputation does not match Batch 1 median")

    def test_electrochemical_correlation(self):
        """Verify the physical correlation between dq_var and log10(cycle_life) is < -0.90."""
        r = np.corrcoef(self.df["dq_var"], self.df["log10_cycle_life"])[0, 1]
        self.assertLess(r, -0.90, f"Expected r < -0.90, got r = {r:.4f}")
        self.assertGreater(r, -0.98, f"Correlation r = {r:.4f} is outside expected range")

if __name__ == "__main__":
    unittest.main()
