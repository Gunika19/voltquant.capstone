"""Filesystem paths for the vq data pipeline."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = PROJECT_ROOT / "data"
RAW = DATA_ROOT / "01_raw"

RAW_NASA = RAW / "nasa"
RAW_EV_DEGRADATION = RAW / "ev_degradation" / "ev_battery_degradation_v1.csv"
RAW_EV_TR = RAW / "ev_charging_tr" / "EV_Battery_Charging_TR_Dataset_with_Notes.csv"

QUARANTINE_NOTE = """
data/01_raw/_quarantine/ holds a pre-cleaned NASA metadata file
(Battery_Data_Cleaned.csv) whose start_time column was dropped during an
earlier cleaning pass. It is kept only for provenance, not for use in the
pipeline, so no path constant points to it — do not add one.
"""
