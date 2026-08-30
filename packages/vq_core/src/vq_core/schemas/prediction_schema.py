"""Schema definition representing the standardized output of VoltQuant models.

This schema forms a clean interface boundary between the ML prognosis pipeline
and downstream systems (FPGA + AES-128 cryptographic packaging + cybersecurity alert nodes).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BatteryAlertSchema:
    """Standardized output container for battery anomaly predictions.
    
    Fields:
        battery_id: The identifier of the battery (e.g. B0005)
        timestamp_s: The relative elapsed time of the measurement in seconds
        discharge_cycle_index: The index of the current discharge cycle
        predicted_state: The classified state: 0 = Normal, 1 = Warning, 2 = Runaway
        risk_probability: The probability/confidence score associated with the runaway state
        model_version: The model architecture version generating the prediction (e.g. VQC_v2.0)
    """
    battery_id: str
    timestamp_s: float
    discharge_cycle_index: int
    predicted_state: int
    risk_probability: float
    model_version: str

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())
