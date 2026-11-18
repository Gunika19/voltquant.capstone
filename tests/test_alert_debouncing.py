"""VoltQuant Test Suite: Multi-Tier State Machine & K=3 Debouncing Filter
Tests:
  1. Injected 1-cycle and 2-cycle noise/regeneration spikes do NOT trigger Tier 3 alarm (0.00% False Alarms).
  2. Persistent degradation across >= 3 consecutive cycles successfully triggers Tier 3 Egress alarm.
  3. Latency is deterministically exactly (K - 1) = 2 cycles.
"""

import unittest
import numpy as np

class DebouncingStateMachine:
    def __init__(self, k_persistence: int = 3, threshold: float = 0.60):
        self.k = k_persistence
        self.threshold = threshold
        self.consecutive_hits = 0
        self.state = 0  # 0: Normal, 1: Advisory, 2: Caution, 3: Critical Egress

    def update(self, raw_probability: float) -> int:
        if raw_probability >= self.threshold:
            self.consecutive_hits += 1
        else:
            self.consecutive_hits = 0

        if self.consecutive_hits == 0:
            self.state = 0
        elif self.consecutive_hits == 1:
            self.state = 1
        elif self.consecutive_hits == 2:
            self.state = 2
        else:
            self.state = 3
        return self.state

class TestAlertDebouncing(unittest.TestCase):
    def test_transient_noise_suppression(self):
        """Verify that isolated 1-cycle and 2-cycle anomalous spikes never reach Tier 3."""
        sm = DebouncingStateMachine(k_persistence=3)
        
        # Stream 50 healthy cycles
        for _ in range(50):
            state = sm.update(0.15)
            self.assertEqual(state, 0)

        # Inject 1-cycle spurious capacity regeneration spike
        state = sm.update(0.85)
        self.assertEqual(state, 1)  # Only Tier 1 Advisory (logged internally)

        # Returns to normal
        state = sm.update(0.15)
        self.assertEqual(state, 0)

        # Inject 2-cycle noise spike
        sm.update(0.80)  # Tier 1
        state = sm.update(0.82)  # Tier 2 Caution
        self.assertEqual(state, 2)
        
        # Falls back to normal
        state = sm.update(0.10)
        self.assertEqual(state, 0, "Failed to reset to Tier 0 after 2-cycle spike")

    def test_persistent_failure_qualification(self):
        """Verify that genuine persistent degradation triggers Tier 3 on cycle 3."""
        sm = DebouncingStateMachine(k_persistence=3)
        history = []
        for p in [0.20, 0.22, 0.75, 0.78, 0.82, 0.85]:
            history.append(sm.update(p))

        self.assertEqual(history, [0, 0, 1, 2, 3, 3])
        # Latency check: first trigger was at index 2 (hit 1), confirmed at index 4 (hit 3) -> exactly 2 cycles delay
        self.assertEqual(history[4], 3)

if __name__ == "__main__":
    unittest.main()

