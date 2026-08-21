"""Clean the synthetic ev_charging_tr dataset for hardware-interface demo use.

Reads data/01_raw/ev_charging_tr/EV_Battery_Charging_TR_Dataset_with_Notes.csv,
writes data/03_features/ev_charging_tr_clean.parquet.

This dataset is AUXILIARY, not training data. Per CLAUDE.md: EventFlag is a
deterministic binning of TR_Probability at cut points 0.40/0.70/0.90
(verified in s01_audit_raw.py), and TR_Probability is ~96% linear in
MaxTemp_C and SOC_% alone -- StateOfHealth_%, pack/cell voltage, charge
power, and BMS_Status carry negligible signal (|r| < 0.02). Its role here is
a synthetic sanity check and hardware-interface demo -- proving the pipeline
can ingest, clean, and encode a charger-telemetry-shaped table end to end --
not evidence of real-world generalisation, and it must never be used as a
comparative benchmark between model architectures (see the printed reminder
at the end of main()).
"""

from __future__ import annotations

import pandas as pd

from vq_core.guards import assert_no_label_leakage, assert_source_columns_dropped
from vq_core.paths import PROJECT_ROOT, RAW_EV_TR
from vq_core.schema import DROP_ON_LOAD, TR_LABEL_BINARY
from vq_data.s01_audit_raw import TR_TIMESTAMP_FORMAT

OUTPUT_PATH = PROJECT_ROOT / "data" / "03_features" / "ev_charging_tr_clean.parquet"
SOURCE_KEY = "ev_charging_tr"

MOHM_PER_OHM = 1000.0

# OK < Warning < Critical -- increasing severity, the only sensible order
# for a BMS health status.
BMS_STATUS_ORDER = ["OK", "Warning", "Critical"]

# Not stated in any source documentation for this synthetic dataset -- this
# is the logical lifecycle of a charging session (negotiate the connection,
# agree charge parameters, bulk-charge, taper at the end), inferred from the
# stage names themselves, not verified against a spec. Flagged here as an
# assumption, not a fact.
CHARGING_STAGE_ORDER = ["Handshake", "Parameter_Config", "Recharge", "End_Of_Charge"]


def section(title: str) -> None:
    print()
    print(f"=== {title} ===")


def load_raw() -> pd.DataFrame:
    section("Load")
    df = pd.read_csv(RAW_EV_TR)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format=TR_TIMESTAMP_FORMAT)
    print(f"input shape: {df.shape}")
    return df


def drop_leaky_and_setpoint_columns(df: pd.DataFrame) -> pd.DataFrame:
    section("Drop leaky / setpoint columns")
    columns = DROP_ON_LOAD[SOURCE_KEY]
    print(
        f"dropping {columns}: TR_Probability and Notes are direct label leaks (Notes narrates the "
        f"EventFlag outcome in prose, TR_Probability *is* the label pre-binning); DemandVoltage_V and "
        f"DemandCurrent_A are charger setpoints, not measurements of the cell's own response."
    )
    return df.drop(columns=columns)


def encode_event_flag(df: pd.DataFrame) -> pd.DataFrame:
    section("Encode EventFlag")
    df["event_flag_raw"] = df["EventFlag"]
    df["EventFlag"] = df["EventFlag"].map(TR_LABEL_BINARY)
    print(
        "EventFlag mapped Watch/Warning -> 0, Alarm/Runaway -> 1 (vq_core.schema.TR_LABEL_BINARY). "
        "This 0/1 split follows the verified TR_Probability cut at 0.70 (Alarm starts at 0.700, Warning "
        "tops out at 0.698 -- see s01_audit_raw.py), not an arbitrary threshold. event_flag_raw keeps the "
        "original 4-level string for traceability."
    )
    assert df["EventFlag"].isna().sum() == 0, "EventFlag mapping produced nulls -- unmapped category present"
    return df


def add_step_index(df: pd.DataFrame) -> pd.DataFrame:
    section("Sort and index steps within cell")
    df = df.sort_values(["CellID", "Timestamp"]).reset_index(drop=True)
    df["step_index"] = df.groupby("CellID").cumcount()
    return df


def convert_units_and_types(df: pd.DataFrame) -> pd.DataFrame:
    section("Unit conversion and type encoding")
    df["internal_resistance_ohm"] = df["InternalResistance_mOhm"] / MOHM_PER_OHM
    print("internal_resistance_ohm = InternalResistance_mOhm / 1000 (unit consistency with NASA's Re/Rct, in ohms)")

    df["MoistureDetected"] = df["MoistureDetected"].astype(int)

    df["BMS_Status"] = pd.Categorical(df["BMS_Status"], categories=BMS_STATUS_ORDER, ordered=True)
    df["ChargingStage"] = pd.Categorical(df["ChargingStage"], categories=CHARGING_STAGE_ORDER, ordered=True)
    print(f"BMS_Status ordered categorical: {BMS_STATUS_ORDER}")
    print(f"ChargingStage ordered categorical: {CHARGING_STAGE_ORDER}")

    assert df["BMS_Status"].isna().sum() == 0, "BMS_Status has a value outside BMS_STATUS_ORDER"
    assert df["ChargingStage"].isna().sum() == 0, "ChargingStage has a value outside CHARGING_STAGE_ORDER"
    return df


def report(df: pd.DataFrame, input_shape: tuple[int, int]) -> None:
    section("Report")
    print(f"input shape: {input_shape}")
    print(f"output shape: {df.shape}")

    print("\nper-cell step counts:")
    print(df.groupby("CellID")["step_index"].max().add(1).describe().to_string())

    print("\nbinary class balance (EventFlag, project encoding):")
    counts = df["EventFlag"].value_counts().sort_index()
    for label, count in counts.items():
        print(f"  {label}: {count} ({100 * count / len(df):.1f}%)")

    dropped = DROP_ON_LOAD[SOURCE_KEY]
    still_present = [c for c in dropped if c in df.columns]
    print(f"\ndropped columns confirmed gone: {still_present == []} (checked {dropped})")


def run_guards(df: pd.DataFrame) -> None:
    section("Guards")
    assert_source_columns_dropped(df, SOURCE_KEY)
    print(f"assert_source_columns_dropped({SOURCE_KEY!r}): PASSED")
    assert_no_label_leakage(df)
    print("assert_no_label_leakage: PASSED")


def print_synthetic_reminder() -> None:
    section("REMINDER: synthetic auxiliary data")
    print(
        "ev_charging_tr is SYNTHETIC. Its label (EventFlag / TR_Probability) is a deterministic\n"
        "binning of TR_Probability, which is itself a near-linear function of MaxTemp_C and SOC_%\n"
        "jointly (R^2 = 0.956, verified by direct OLS fit -- MaxTemp_C alone already gives R^2 = 0.52,\n"
        "per CLAUDE.md/s01_audit_raw.py). StateOfHealth_%, pack/cell voltage, charge power, and\n"
        "BMS_Status carry negligible signal (|r| < 0.02). A model that recovers this label is\n"
        "recovering a near-linear function of two columns baked into the generator, not detecting\n"
        "thermal runaway. This dataset CANNOT discriminate between model architectures and MUST NOT\n"
        "be used as a comparative benchmark -- its role is a synthetic sanity check and\n"
        "hardware-interface demo only."
    )


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = load_raw()
    input_shape = df.shape

    df = drop_leaky_and_setpoint_columns(df)
    df = encode_event_flag(df)
    df = add_step_index(df)
    df = convert_units_and_types(df)

    report(df, input_shape)
    run_guards(df)

    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nwrote {OUTPUT_PATH} ({df.shape[0]} rows, {df.shape[1]} columns)")

    print_synthetic_reminder()


if __name__ == "__main__":
    main()
