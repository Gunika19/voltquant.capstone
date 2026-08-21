## Known dataset properties

- `ev_charging_tr` is synthetically generated. `EventFlag` is a deterministic binning of
  `TR_Probability` at cut points 0.40 / 0.70 / 0.90 (non-overlapping ranges, verified in s01).
  `TR_Probability` is approximately a function of temperature and SOC (MaxTemp alone gives
  R² = 0.52); `StateOfHealth_%`, pack and cell voltage, charge power, and `BMS_Status` carry
  negligible signal (|r| < 0.02). Role: synthetic sanity check and hardware-interface demo
  data only — not used for training, and not evidence of real-world generalisation.

- `ev_battery_degradation` has 4 positives in 10,000 rows (0.04%). `Battery_Status` is not an
  SoH threshold — cells at 74% SoH are still labelled Healthy. Role: descriptive context
  only; cannot support a supervised claim.

- NASA: time-to-2.7V correlates with `soh` at **r ≈ 0.999** within a fixed
  `(battery, ambient_temperature, load_level)` stratum (`s05b_check_sampling_confound.py`).
  Physical reason: at constant load current, discharged capacity is current × time, so time to
  reach a fixed voltage cutoff is discharged capacity up to a constant multiplier — it is an
  *alternative measurement* of the same quantity `soh` is built from, not an independent
  predictor. The one apparent counterexample, B0042 at r = −0.42, is **Simpson's paradox**: that
  cell recorded under two different `(ambient_temperature, load_level)` strata, and the pooled
  correlation flips sign purely from mixing them — within each stratum it is r = 1.0. This is a
  reportable result for the paper's deployability argument (a near-perfect, physically-grounded
  precursor signal exists in this data) but `time_to_cutoff_s` / `time_to_2v7_s` must never enter
  a feature matrix — see `vq_core.schema.TIME_TO_CUTOFF_IS_LABEL_DERIVED` and
  `LABEL_ONLY_FEATURES`.

## Guardrails (vq_core.guards)

- `assert_source_columns_dropped(df, source_key)` — raises if any column listed in
  `DROP_ON_LOAD[source_key]` (see `vq_core.schema`) is still present in `df`. This only knows
  about the synthetic EV sources' direct leaks and setpoint columns (`TR_Probability`, `Notes`,
  etc.) — it says nothing about `capacity`/`re`/`rct`, so it cannot catch the NASA leakage risk.
  Formerly named `assert_no_leakage`.
- `assert_no_label_leakage(df)` — the real guard against the project's actual leakage risk.
  Raises `LabelLeakageError` naming every column in `df` that matches (case-insensitively)
  `vq_core.schema.LABEL_ONLY_FEATURES` — `capacity`, `capacity_ref`, `soh`, `soh_pct`, `re`,
  `rct`, `time_to_cutoff_s`, `time_to_2v7_s`, in any casing. These columns exist only to
  construct the SOH label (directly, or as an alternative measurement of it — see
  `TIME_TO_CUTOFF_IS_LABEL_DERIVED` above) and must never appear in a model input matrix. Call
  this on any table about to be handed to a model.
- `assert_model_input_ready(df, feature_columns)` — runs `assert_no_label_leakage`, then raises
  if any of `feature_columns` is missing from `df`, and warns (doesn't raise) if `df` has columns
  present but not declared in `feature_columns`. This is the one to call right before training.
- `assert_grouped_split(df, split_column, group_column)` — raises if any `group_column` value
  (e.g. `battery_id`) appears under more than one `split_column` value (e.g. both "train" and
  "test"). Call this after any train/test split to guarantee the split is grouped, not row-wise.

**`data/02_intermediate/nasa_discharge_cycles.parquet` is a label-construction table** — it
intentionally contains `capacity`, `capacity_ref`, `soh`, and `soh_pct` (see the module docstring
of `s04_build_discharge_table.py`). It must never be passed to a model directly; running
`assert_no_label_leakage` against it is expected to raise. Derive a features table from it first,
then guard that downstream table instead.
