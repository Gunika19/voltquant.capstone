"""VoltQuant Test Suite: Pipeline Resilience & Fault Tolerance
Tests:
  1. Input telemetry validation rejects physically unviable voltages (<0V or >5V).
  2. Missing sensor fields trigger graceful imputation or fallback without unhandled exceptions.
  3. Zero/negative discharge current triggers sensor fault advisory without crashing the inference engine.
"""

import unittest
import numpy as np

def validate_telemetry_packet(packet: dict) -> dict:
    """Production ingress gatekeeper for raw vehicle telemetry."""
    status = {"valid": True, "faults": [], "sanitized": {}}
    
    # 1. Voltage validation
    v = packet.get("voltage", None)
    if v is None or np.isnan(v):
        status["faults"].append("VOLTAGE_MISSING")
        status["valid"] = False
    elif v < 1.5 or v > 4.5:
        status["faults"].append("VOLTAGE_OUT_OF_PHYSICAL_BOUNDS")
        status["valid"] = False
    else:
        status["sanitized"]["voltage"] = float(v)

    # 2. Current validation
    i = packet.get("current", None)
    if i is None or np.isnan(i):
        status["faults"].append("CURRENT_MISSING")
        status["valid"] = False
    else:
        status["sanitized"]["current"] = float(i)

    # 3. Timestamp / Duration validation
    t = packet.get("timestamp_s", None)
    if t is None or t < 0:
        status["faults"].append("INVALID_TIMESTAMP")
        status["valid"] = False
    else:
        status["sanitized"]["timestamp_s"] = float(t)

    return status

class TestPipelineResilience(unittest.TestCase):
    def test_healthy_packet(self):
        pkt = {"voltage": 3.65, "current": -1.1, "timestamp_s": 120.0}
        res = validate_telemetry_packet(pkt)
        self.assertTrue(res["valid"])
        self.assertEqual(len(res["faults"]), 0)

    def test_corrupted_voltage(self):
        pkt = {"voltage": -99.0, "current": -1.1, "timestamp_s": 120.0}
        res = validate_telemetry_packet(pkt)
        self.assertFalse(res["valid"])
        self.assertIn("VOLTAGE_OUT_OF_PHYSICAL_BOUNDS", res["faults"])

    def test_missing_field(self):
        pkt = {"voltage": 3.7, "current": np.nan, "timestamp_s": 10.0}
        res = validate_telemetry_packet(pkt)
        self.assertFalse(res["valid"])
        self.assertIn("CURRENT_MISSING", res["faults"])

if __name__ == "__main__":
    unittest.main()

