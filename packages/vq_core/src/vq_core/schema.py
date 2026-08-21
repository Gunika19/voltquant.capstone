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
    "time_to_cutoff_s",
    "time_to_2v7_s",
    # s06_build_labels.py: the early-warning label itself and its direct
    # derivative. cycles_to_crossing is signed distance to the sustained
    # crossing cycle -- it encodes exactly how close a row is to the label
    # transition, i.e. future knowledge, same reasoning as
    # TIME_TO_CUTOFF_IS_LABEL_DERIVED above.
    "label",
    "cycles_to_crossing",
]

# s05b_check_sampling_confound.py found time-to-2.7V correlates with soh at
# r ~ 0.999 within a (battery, ambient_temperature, load_level) stratum (the
# one apparent exception, B0042 at r=-0.42, is Simpson's paradox: it recorded
# under two strata, and resolves to r=1.0 within each -- see CLAUDE.md). This
# is not a coincidence and not an independent predictor: at fixed load
# current, capacity discharged is current x time, so time-to-a-fixed-voltage-
# cutoff is discharged capacity up to a constant. It is an alternative
# measurement of the same physical quantity soh is built from, not new
# information -- report it, never train on it. Hence it lives in
# LABEL_ONLY_FEATURES (as time_to_cutoff_s / time_to_2v7_s) rather than as an
# ordinary feature.
TIME_TO_CUTOFF_IS_LABEL_DERIVED = True

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

# s06_build_labels.py: early-warning label scheme.
#
# 0.80 is the conventional SOH end-of-life threshold in battery prognostics
# (paired with the 30% fade / 2 Ahr -> 1.4 Ahr rating already documented
# above as NOMINAL_CAPACITY_AH -- 1.4/2.0 = 0.70 is the nameplate fade
# limit, but 0.80 is the more commonly cited *early*-warning cutoff in the
# prognostics literature, giving room for a warning window before nameplate
# end-of-life is reached).
SOH_EOL_THRESHOLD = 0.80

# s06a_crossing_budget.py swept lead times 5/10/15/20/30/50 and found 20 gives
# the best three-class balance among the lead times with the fewest cells
# whose pre-crossing history is too short to fill the window -- see that
# script's `recommend()` output.
WARNING_LEAD_CYCLES = 20

# s06a found 4 of 16 naive crossing cells (B0036/B0049/B0054/B0055) cross at
# discharge_cycle_index=0 purely because their first valid cycle is a single
# noisy low reading that rebounds above threshold on the very next cycle --
# not sustained decline. Requiring the crossing to hold for 3 consecutive
# valid cycles after the initial dip is the smallest window that resolves
# all 4 known artifacts in this dataset (verified in s06_build_labels.py's
# naive-vs-sustained report) without being so long it erases genuine
# near-end-of-life crossings that have fewer than 3 cycles of history left
# to confirm against.
CROSSING_PERSISTENCE_CYCLES = 3

# Early-warning three-class label. 2 (runaway) starts at the sustained
# crossing cycle itself, not one cycle after -- see LABELS use in
# s06_build_labels.py.
LABELS = {0: "normal", 1: "warning", 2: "runaway"}
