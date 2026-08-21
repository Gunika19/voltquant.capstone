# capstone

A hybrid quantum-classical machine learning (HQML) pipeline for early-warning
prediction of Li-ion battery thermal runaway, paired with a planned
FPGA-based AES-128 hardware security module (HSM) for issuing tamper-proof
alerts. The data-engineering pipeline builds a leakage-guarded, three-class
early-warning dataset from NASA's Li-ion battery discharge cycles; the
classical and quantum modelling tracks, the evaluation harness, and the
hardware alerting path have not been started yet.

## Status

- **Data pipeline: complete through `s10`.** Ten scripts (`s01`-`s10`, plus
  three read-only diagnostics `s05b`, `s06a`, `s08b`) take the raw NASA
  discharge data through auditing, SOH-label construction, feature
  engineering, early-warning labelling, sequencing, scaling, and a
  feature-redundancy/qubit-budget analysis, ending in a scaled,
  train/val/test-split, leakage-checked model-input dataset. Two auxiliary
  EV sources are also cleaned/summarised (`s09`, `s10`).
- **Classical modelling (`vq_classical`, an LSTM benchmark): not started.**
  No package exists yet.
- **Quantum modelling (`vq_quantum`, a VQC): not started.** No package
  exists yet; `s08b_feature_redundancy.py` has done preparatory analysis
  (which features fit an 8-qubit budget) but no VQC has been built or
  trained.
- **Evaluation harness (`vq_eval`): not started.**
- **FPGA AES-128 HSM hardware path: not started.**

## Data

Three raw sources feed this repository, and only one of them is used for
training.

**NASA Li-ion battery dataset — the training corpus.** Discharge cycles
from real Li-ion cells run to end-of-life under varying ambient
temperature and load current. After scoping to cells with a clean,
unambiguous capacity trend (verdict `CLEAN` or `STRATIFIED` in
`s04_build_discharge_table.py`, restricted to the dominant
temperature/load stratum for the one `STRATIFIED` cell) and building the
early-warning label (`s06_build_labels.py`), the final model-input dataset
(`data/04_model_input/`, documented in `DATASET_CARD.md`) is:

- **23 usable cells**, split by battery so no cell crosses a split boundary:
  14 cells / 10,607 windows (train), 3 cells / 1,372 windows (val), 6 cells
  / 6,385 windows (test) — **18,364 windows total**.
- **22 features** per timestep, auto-detected and reviewed for leaks
  (`s07_build_sequences.py`), standardised on train statistics only.
- Splits are grouped by `battery_id` and stratified by whether the cell
  ever reaches `runaway`, so no split is left without positive examples.

**`ev_charging_tr` (synthetic charger telemetry) — documented, not
trained on.** Its `EventFlag` label is a deterministic binning of
`TR_Probability` at fixed cut points, and `TR_Probability` itself is a
near-linear function of `MaxTemp_C` and `SOC_%` jointly (R² = 0.956,
verified by direct OLS fit in `s09_clean_ev_charging_tr.py`) — a model that
recovers this label is recovering that near-linear function, not detecting
thermal runaway. Its role is a synthetic sanity check and
hardware-interface demo: proving the pipeline can ingest, clean, and encode
a charger-telemetry-shaped table end to end.

**`ev_battery_degradation` (fleet telemetry) — documented, not trained
on.** Its `Battery_Status` label separates cleanly on `SoH_Percent` with no
overlap (Healthy 80.17-100%, Replace Required 73.74-78.96%), so the label
is derivable from a feature already in the table and adds no independent
information — and with only 4 positive rows in 10,000, it could not support
a supervised claim even if it weren't. Its role is descriptive context for
the paper's dataset section (`reports/ev_degradation_summary.md`).

## Label definition

Three-class early-warning label, assigned per discharge cycle and inherited
by every measurement/window within that cycle (`vq_core.schema`,
`s06_build_labels.py`):

- **runaway (2)**: `discharge_cycle_index >= sustained_crossing_index`
- **warning (1)**: within `WARNING_LEAD_CYCLES` (20) cycles before the
  sustained crossing
- **normal (0)**: everything else, including every cycle of a cell that
  never has a sustained crossing

`sustained_crossing_index` is the first cycle where `soh < SOH_EOL_THRESHOLD`
(0.80) **and it stays below that threshold for the next
`CROSSING_PERSISTENCE_CYCLES` (3) consecutive valid cycles** — not the
naive first-crossing point. The persistence requirement exists because 4 of
16 naive-crossing cells crossed at their very first recorded cycle purely
from a single noisy low reading that rebounded the next cycle, not sustained
decline.

## Repository layout

```
packages/
  vq_core/    shared, dependency-light utilities: column-safety constants
              and guard functions (vq_core.schema, vq_core.guards), the
              FULL_FEATURES / VQC_FEATURES feature-set constants
              (vq_core.features), and filesystem paths (vq_core.paths).
  vq_data/    the s01-s10 data pipeline scripts (this is what's built).
envs/         pinned, mutually exclusive dependency sets: data.txt
              (the pipeline above), classical.txt (TensorFlow), and
              quantum.txt (PennyLane) -- classical and quantum pin
              incompatible numpy/protobuf ranges, hence two separate files.
.github/      CI workflow (pytest + ruff, no data required).
reports/      ev_degradation_summary.md (tracked); reports/figures/ is
              gitignored, regenerated locally by the pipeline scripts.
data/         gitignored. 01_raw/ (inputs you provide), 02_intermediate/,
              03_features/, 04_model_input/ (pipeline outputs).
CLAUDE.md     verified dataset properties and gotchas discovered while
              building the pipeline.
Makefile      env-data / env-classical / env-quantum / test / lint / clean.
```

`vq_classical` and `vq_quantum` don't exist yet (see Status). When built,
the architecture is that they never import each other directly — they
communicate only through the model-input artifacts in `data/04_model_input/`
and shared utilities in `vq_core`.

## Getting started

1. Clone the repo, and install [uv](https://docs.astral.sh/uv/) if you
   don't have it.
2. Build the data-pipeline environment: `make env-data`. This creates
   `.venv-data` and installs the pins in `envs/data.txt`, including both
   local packages in editable mode.
3. Obtain the raw data (all gitignored, none of it ships in this repo) and
   place it under `data/01_raw/`:
   - NASA Li-ion battery discharge data at `data/01_raw/nasa/` —
     `metadata.csv` plus a `data/` subdirectory of per-cycle CSVs.
   - `data/01_raw/ev_charging_tr/EV_Battery_Charging_TR_Dataset_with_Notes.csv`
   - `data/01_raw/ev_degradation/ev_battery_degradation_v1.csv`
4. Run the pipeline in order:
   ```
   .venv-data/bin/python -m vq_data.s01_audit_raw
   .venv-data/bin/python -m vq_data.s02_verify_nasa
   .venv-data/bin/python -m vq_data.s03_diagnose_capacity_trend
   .venv-data/bin/python -m vq_data.s04_build_discharge_table
   .venv-data/bin/python -m vq_data.s05_build_measurement_features
   .venv-data/bin/python -m vq_data.s06_build_labels
   .venv-data/bin/python -m vq_data.s07_build_sequences
   .venv-data/bin/python -m vq_data.s08_scale_and_finalise
   .venv-data/bin/python -m vq_data.s09_clean_ev_charging_tr
   .venv-data/bin/python -m vq_data.s10_summarise_ev_degradation
   ```
   `s01`-`s03` and the diagnostics `s05b_check_sampling_confound`,
   `s06a_crossing_budget`, and `s08b_feature_redundancy` are read-only —
   they print reports and figures but write no data files, and can be run
   any time after the files they read exist.
5. `make test` runs the test suite; `make lint` runs ruff.

`make env-classical` and `make env-quantum` build the (currently unused)
environments for the classical and quantum tracks once that code exists.

## Known limitations

- **8 `ANOMALOUS`-verdict NASA cells are excluded entirely** — no held-out
  edge-case set covering them has been built yet.
- **B0045 has no baseline** capacity reading anywhere in its recorded
  history and is excluded as a result.
- **The 80% SOH crossing point is sensitive to how `capacity_ref` is
  estimated**: across four candidate estimators, the crossing point shifts
  by more than 5 cycles for 12 of 34 batteries. This dataset uses the
  top-5-median estimator throughout; a different choice would move some
  cells' labels.
- **Sampling interval is a per-cycle covariate, not a constant** — it
  varies 9.3-20.5s across the dataset, including within a single battery's
  own recorded history. It's included as a feature so downstream models
  can condition on it.
- **No resampling has been applied.** Class balance is reported as
  observed; any class-balance correction is a training-time decision that
  hasn't been made, and would have to be fit on the train split only.
- **Label-derived columns are barred from feature matrices by
  construction**, not by convention — see Data integrity guarantees below.

## Data integrity guarantees

Several leakage and split risks in this pipeline are enforced by guard
functions and tests, not left to convention:

- `assert_no_label_leakage` raises `LabelLeakageError` if a table about to
  be used as model input contains any column in
  `vq_core.schema.LABEL_ONLY_FEATURES` — `capacity`, `soh`, `re`, `rct`,
  `time_to_cutoff_s`, `label`, `cycles_to_crossing`, and their case
  variants. This catches both direct label columns and columns that are an
  *alternative measurement* of the label rather than an independent
  feature (e.g. `time_to_cutoff_s` correlates with `soh` at r ≈ 0.999
  within a fixed condition — it's derived from the same discharged-capacity
  quantity, not new information).
- `assert_source_columns_dropped` raises if a synthetic source's known
  direct leaks or setpoint columns (e.g. `ev_charging_tr`'s
  `TR_Probability` and `Notes`) are still present after cleaning.
- `assert_grouped_split` raises if any `battery_id` appears under more than
  one split — the guarantee behind "23 usable cells, no cell in more than
  one split" above.
- The feature scaler is fit on the train split only and applied unchanged
  to val/test, because fitting on the full set would leak test-split
  statistics into the transform applied to train.
- These guards are exercised by an automated test suite
  (`packages/vq_core/tests/`), run via `make test` and in CI on every push
  and pull request — not just called optionally from pipeline scripts.
