"""VoltQuant Unified Test Runner

Runs all unit, integration, and resilience tests across the VoltQuant codebase:
  - test_data_integrity.py
  - test_model_reproducibility.py
  - test_alert_debouncing.py
  - test_pipeline_resilience.py
"""

import sys
import unittest
import time
from pathlib import Path

def run_suite():
    print("=" * 75)
    print("VOLTQUANT AUTOMATED QUALITY & REPRODUCIBILITY TEST SUITE")
    print("=" * 75)

    test_dir = Path(__file__).resolve().parent
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(test_dir), pattern="test_*.py")

    start_time = time.time()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    elapsed = time.time() - start_time

    print("=" * 75)
    print(f"Tests Run     : {result.testsRun}")
    print(f"Failures      : {len(result.failures)}")
    print(f"Errors        : {len(result.errors)}")
    print(f"Success Rate  : {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun) * 100:.1f}%")
    print(f"Execution Time: {elapsed:.2f} seconds")
    print("=" * 75)

    if not result.wasSuccessful():
        sys.exit(1)

if __name__ == "__main__":
    run_suite()

