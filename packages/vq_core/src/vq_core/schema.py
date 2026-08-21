"""Column-level safeguards for the vq data pipeline."""

DROP_ON_LOAD = {
    "ev_charging_tr": [
        "TR_Probability",
        "Notes",
        "DemandVoltage_V",
        "DemandCurrent_A",
    ],
    "ev_degradation": [
        "Vehicle_ID",
        "Car_Model",
        "Battery_Type",
        "Driving_Style",
    ],
}

# ev_charging_tr: this ordering is not an assumption -- it is a verified fact.
# EventFlag is a deterministic binning of TR_Probability into contiguous,
# non-overlapping ranges (checked against the raw data in s01_audit_raw.py):
#   Watch    0.247-0.392
#   Warning  0.406-0.698
#   Alarm    0.700-0.899
#   Runaway  0.902-1.000
TR_EVENT_FLAG_ORDER = ["Watch", "Warning", "Alarm", "Runaway"]

# Empirical cut points that reproduce the binning above.
TR_PROBABILITY_BINS = [0.40, 0.70, 0.90]


def event_flag_from_probability(p: float) -> str:
    """Reproduce EventFlag from TR_Probability using the verified cut points."""
    low, mid, high = TR_PROBABILITY_BINS
    if p < low:
        return "Watch"
    if p < mid:
        return "Warning"
    if p < high:
        return "Alarm"
    return "Runaway"


# Binary risk label used elsewhere in the project (0 = normal, 1 = warning/risk).
# The split falls at the 0.70 probability cut, not an arbitrary midpoint:
# Watch/Warning sit below it, Alarm/Runaway sit at or above it.
TR_LABEL_BINARY = {
    "Watch": 0,
    "Warning": 0,
    "Alarm": 1,
    "Runaway": 1,
}

# Columns that exist only to construct a label (capacity/impedance measurements
# and everything derived from them in s04_build_discharge_table.py). None of
# these may appear in a model input matrix -- that's what makes them
# label-only rather than ordinary features. Matched case-insensitively in
# vq_core.guards, so both the raw NASA casing (Capacity, Re, Rct) and the
# lowercase columns written to nasa_discharge_cycles.parquet are caught.
LABEL_ONLY_FEATURES = [
    "capacity",
    "capacity_ref",
    "soh",
    "soh_pct",
    "re",
    "rct",
    "Capacity",
    "Re",
    "Rct",
]
