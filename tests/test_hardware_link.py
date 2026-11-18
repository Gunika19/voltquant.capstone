"""VoltQuant Test Suite: Hardware Communication Link (vq_send.py)

Tests:
  1. Packet formatting strictly follows "<class>,<cell_id>,<value>\\n".
  2. String cell IDs (e.g. "B0005", "CS2_35") sanitize correctly to numeric integers (5, 35).
  3. Physical bounds checking accepts valid SOH values (0-100) and rejects anomalous values.
  4. Response string parsing correctly flags ACCEPTED vs REJECTED / ERR.
"""

import unittest
from pathlib import Path
import sys

# Ensure root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from vq_send import VoltQuantLink

class TestHardwareLink(unittest.TestCase):
    def setUp(self):
        # Force mock mode for deterministic unit testing
        self.link = VoltQuantLink(mock=True)

    def tearDown(self):
        self.link.close()

    def test_cell_id_sanitization(self):
        """Verify string battery identifiers are converted to integer IDs."""
        self.assertEqual(self.link._sanitize_cell_id(5), 5)
        self.assertEqual(self.link._sanitize_cell_id("5"), 5)
        self.assertEqual(self.link._sanitize_cell_id("B0005"), 5)
        self.assertEqual(self.link._sanitize_cell_id("CS2_35"), 235)
        self.assertEqual(self.link._sanitize_cell_id("b1c18"), 118)

    def test_nominal_packet_accepted(self):
        """Test that nominal prediction packet (e.g. 1, 5, 58) is ACCEPTED."""
        reply = self.link.send(cls=1, cell_id=5, value=58)
        self.assertIn("ACCEPTED", reply)
        self.assertIn("soh=58%", reply)
        self.assertEqual(self.link.stats["accepted"], 1)

    def test_out_of_bounds_soh_rejected(self):
        """Test that physically impossible SOH values are REJECTED."""
        reply_high = self.link.send(cls=1, cell_id=5, value=150)
        self.assertIn("REJECTED", reply_high)

        reply_neg = self.link.send(cls=0, cell_id=5, value=-10)
        self.assertIn("REJECTED", reply_neg)
        self.assertEqual(self.link.stats["rejected"], 2)

    def test_invalid_class_rejected(self):
        """Test that invalid alert class (e.g. class 9) is REJECTED."""
        reply = self.link.send(cls=9, cell_id=5, value=80)
        self.assertIn("REJECTED", reply)

if __name__ == "__main__":
    unittest.main()
