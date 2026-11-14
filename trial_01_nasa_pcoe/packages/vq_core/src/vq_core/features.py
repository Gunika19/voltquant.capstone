"""Feature-set constants shared between the classical and quantum model input pipelines.

Both lists name columns of data/03_features/nasa_labelled_measurements.parquet
(post s07_build_sequences.py's Time-drop -- see CLAUDE.md for why Time itself
is excluded: it is byte-identical to time_elapsed). Every number cited below
comes from packages/vq_data/src/vq_data/s08b_feature_redundancy.py's
correlation/PCA/practical-test report against the scaled train split; re-run
that script if the underlying data changes and these comments start to drift.
"""

# The complete auto-detected feature set s07_build_sequences.py builds
# sequences from -- numeric columns minus identifiers and label-derived
# columns (see s07's detect_feature_columns). This is the classical model's
# (the vq_classical LSTM's) full input.
FULL_FEATURES = [
    "time_elapsed",
    "Voltage_measured",
    "Current_measured",
    "Temperature_measured",
    "Current_load",
    "Voltage_load",
    "dT_dt",
    "dV_dt",
    "dI_dt",
    "power",
    "roll_mean_temperature_samples",
    "roll_std_temperature_samples",
    "roll_mean_voltage_samples",
    "roll_std_voltage_samples",
    "roll_mean_temperature_60s",
    "roll_std_temperature_60s",
    "roll_mean_voltage_60s",
    "roll_std_voltage_60s",
    "temp_rise",
    "sampling_interval_s",
    "ambient_temperature",
    "load_level",
]

# The 8 features fed to the VQC (its qubit budget is 8, one feature per
# qubit). s08b_feature_redundancy.py's practical test (multinomial logistic
# regression, fit on train / evaluated on val) found this hand-picked set
# beats PCA-8 on both FULL_FEATURES and the 18-feature rolling-mean-reduced
# set on macro F1 (0.666 vs 0.659 / 0.660) -- and by a much wider margin on
# runaway recall specifically (0.501 vs 0.169 / 0.165), the class an
# early-warning system cares about most. Max pairwise |r| within this set of
# 8 is 0.469 (Current_measured <-> dT_dt) -- no internal near-duplicates.
#
# One raw signal per physical quantity, kept:
#   Voltage_measured, Current_measured, Temperature_measured
# First derivatives of the two quantities that move slowly enough for a
# derivative to mean something (temperature, voltage) -- the claimed
# precursor signals:
#   dT_dt, dV_dt
# One smoothed volatility measure per quantity that has one (temperature,
# voltage), using the 60s time-window variant, not the 5-sample one --
# sampling interval varies 9.3-20.5s across cells (see
# s05b_check_sampling_confound.py), so 60s is the physically fixed window
# while "5 samples" is not:
#   roll_std_temperature_60s, roll_std_voltage_60s
# Within-cycle position:
#   time_elapsed
#
# Excluded, and why (all r values from s08b's train-split correlation matrix):
#   - dI_dt: r=0.936 with dV_dt -- the two move almost together at fixed
#     load; dV_dt is kept as the representative.
#   - power: r=0.988 with Current_measured (power = Voltage_measured *
#     Current_measured, and current is roughly constant within a cycle at
#     fixed load, so the product just tracks current almost linearly).
#   - roll_std_temperature_samples / roll_std_voltage_samples: r=0.965 and
#     r=0.960 against their own 60s-window counterparts, which are kept
#     instead (see above).
#   - roll_mean_temperature_samples, roll_mean_temperature_60s,
#     roll_mean_voltage_samples, roll_mean_voltage_60s: r>0.998 against
#     Temperature_measured / Voltage_measured respectively -- exact
#     duplicates of the raw signal, not independent smoothing.
#   - temp_rise: derived from Temperature_measured (temperature minus its
#     own cycle-start value), correlates with both Temperature_measured and
#     dT_dt -- redundant with signal already kept.
#   - Current_load, Voltage_load: setpoints, not measurements of the cell's
#     own response -- Current_load is close to a re-encoding of the
#     load_level stratum, not a degradation signal.
#   - sampling_interval_s, ambient_temperature, load_level: experimental
#     protocol covariates (which rate this row was sampled at, and which
#     temperature/load stratum this cell ran under), not a measurement of
#     battery state -- ambient_temperature also correlates r=0.961 with
#     Temperature_measured, which is kept instead.
VQC_FEATURES = [
    "Voltage_measured",
    "Current_measured",
    "Temperature_measured",
    "dT_dt",
    "dV_dt",
    "roll_std_temperature_60s",
    "roll_std_voltage_60s",
    "time_elapsed",
]
