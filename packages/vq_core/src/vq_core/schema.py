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

# nasa/metadata.csv: these 9 uids store Re/Rct as a Python complex-number
# literal, e.g. "(0.0499-0.0293j)", instead of a plain float. This is not
# malformed data -- impedance is physically a complex quantity (R + jX), and
# these rows are genuine EIS-derived measurements on batteries B0049 and
# B0051. pd.to_numeric(errors="coerce") cannot parse them and would silently
# null out a present measurement; parse with Python's complex() instead
# (verified in s01_audit_raw.py).
RE_RCT_COMPLEX_UIDS = {4268, 4280, 4282, 4292, 4294, 4306, 4316, 4318, 4454}

# NASA PCoE Li-ion battery dataset: cells are rated at 2 Ahr nominal capacity,
# with end-of-life defined as 30% fade (2 Ahr -> 1.4 Ahr) -- stated verbatim
# in data/01_raw/nasa/extra_infos/README_05_06_07_18.txt ("a 30% fade in
# rated capacity (from 2Ahr to 1.4Ahr)"). B0006's observed max Capacity
# (2.014 Ah) is consistent with this rating. Not used as the operative
# capacity_ref (see s04_build_discharge_table.py) -- kept for the
# four-estimator sensitivity check and as a documented limitation.
NOMINAL_CAPACITY_AH = 2.0

# nasa/metadata.csv: batteries with no fresh-state reading anywhere in their
# recorded discharge history, so no capacity_ref (of any definition) can be
# a beginning-of-life reference for them. B0045's first recorded discharge
# (~1.08 Ah) equals its all-time max -- it never shows a higher reading at
# any later cycle either -- while its same-batch, same-start-timestamp
# siblings B0046-B0048 begin at 1.66-1.73 Ah. These batteries get verdict
# NO_BASELINE in s04_build_discharge_table.py rather than a computed SOH.
FRESH_STATE_MISSING = ["B0045"]
