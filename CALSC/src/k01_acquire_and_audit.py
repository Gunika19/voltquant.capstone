"""K01 · Acquire and Audit (READ-ONLY)

Inspects CALCE CS2 cell data:
1. Audits all column headers in cycler logs to verify if ANY temperature channel exists.
2. Audits chronological ordering across multi-file logs (timestamps, dates, cycle indices).
"""

from __future__ import annotations

import os
import re
import sys
import zipfile
from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/01_raw/calce")
ZIP_PATH = RAW_DIR / "CS2_35.zip"
EXTRACT_DIR = RAW_DIR / "CS2_35"

def ensure_unpacked():
    if not ZIP_PATH.exists():
        print(f"Error: {ZIP_PATH} does not exist.")
        sys.exit(1)
    
    if not EXTRACT_DIR.exists():
        print(f"Extracting {ZIP_PATH} to {EXTRACT_DIR}...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
        print("Extraction complete.")
    else:
        print(f"Extracted folder already exists at {EXTRACT_DIR}")

def audit_files():
    ensure_unpacked()
    
    # Collect all files recursively
    files = [p for p in EXTRACT_DIR.rglob("*") if p.is_file() and not p.name.startswith(".")]
    print(f"\nTotal files in archive: {len(files)}")
    
    # Sort files by name initially
    files = sorted(files, key=lambda p: p.name)
    print("\nSample filenames:")
    for f in files[:10]:
        print(f"  {f.name} ({f.stat().st_size:,} bytes)")
    if len(files) > 10:
        print(f"  ... and {len(files)-10} more files.")
        for f in files[-5:]:
            print(f"  {f.name} ({f.stat().st_size:,} bytes)")
            
    # Inspect first file
    sample_file = files[0]
    print(f"\n--- Detailed Inspection of Sample File: {sample_file.name} ---")
    
    # Check extension
    ext = sample_file.suffix.lower()
    print(f"File extension: {ext}")
    
    sheet_names = []
    columns_found = set()
    sample_df = None
    
    if ext in [".xlsx", ".xls"]:
        excel_file = pd.ExcelFile(sample_file)
        sheet_names = excel_file.sheet_names
        print(f"Sheets found: {sheet_names}")
        
        # Read headers of all sheets
        for s in sheet_names:
            df_header = pd.read_excel(excel_file, sheet_name=s, nrows=5)
            print(f"\nSheet '{s}' columns ({len(df_header.columns)}):")
            print(list(df_header.columns))
            columns_found.update(df_header.columns)
            if sample_df is None and len(df_header) > 0:
                sample_df = df_header
    elif ext in [".txt", ".csv"]:
        sample_df = pd.read_csv(sample_file, nrows=5, sep=None, engine='python')
        print(f"\nColumns ({len(sample_df.columns)}):")
        print(list(sample_df.columns))
        columns_found.update(sample_df.columns)
    
    # Critical Check 1: ANY temperature column?
    print("\n==========================================")
    print("CRITICAL AUDIT 1: TEMPERATURE CHANNEL CHECK")
    print("==========================================")
    temp_keywords = ["temp", "t_", "temperature", "thermocouple", "deg", "celsius"]
    matching_temp = [col for col in columns_found if any(kw in str(col).lower() for kw in temp_keywords)]
    if matching_temp:
        print(f"ATTENTION: Potential temperature columns found: {matching_temp}")
    else:
        print("CONFIRMED: ZERO temperature channels found across columns in sample file.")
        print("All columns recorded:")
        for col in sorted(columns_found, key=lambda x: str(x)):
            print(f"  - {col}")
            
    # Critical Check 2: Chronological Ordering Analysis
    print("\n==========================================")
    print("CRITICAL AUDIT 2: CHRONOLOGICAL ORDERING")
    print("==========================================")
    
    # Check timestamps across multiple files
    date_records = []
    print(f"Inspecting file headers and timestamps across files...")
    
    for idx, p in enumerate(files):
        # Read minimal rows to check Start Time / Date / Test Time / Cycle Index
        try:
            if p.suffix.lower() in [".xlsx", ".xls"]:
                xl = pd.ExcelFile(p)
                # target sheet usually 'Channel_1_1' or 'Data' or first sheet
                target_sheet = "Channel_1_1" if "Channel_1_1" in xl.sheet_names else xl.sheet_names[0]
                df_head = pd.read_excel(xl, sheet_name=target_sheet, nrows=2)
            else:
                df_head = pd.read_csv(p, nrows=2, sep=None, engine='python')
                
            start_date_time = None
            start_cycle = None
            
            for col in df_head.columns:
                c_clean = str(col).strip().replace(" ", "_").lower()
                if "date_time" in c_clean or "datetime" in c_clean:
                    start_date_time = df_head[col].iloc[0]
                if "cycle_index" in c_clean or c_clean == "cycle":
                    start_cycle = df_head[col].iloc[0]
                    
            date_records.append({
                "filename": p.name,
                "start_date_time": str(start_date_time),
                "start_cycle": start_cycle,
                "file_size": p.stat().st_size
            })
        except Exception as e:
            date_records.append({
                "filename": p.name,
                "error": str(e)
            })
            
    df_order = pd.DataFrame(date_records)
    print("\nFirst 10 files in alphabetic/lexicographical order:")
    print(df_order[["filename", "start_date_time", "start_cycle"]].head(10).to_string())
    
    print("\nLast 5 files in alphabetic/lexicographical order:")
    print(df_order[["filename", "start_date_time", "start_cycle"]].tail(5).to_string())
    
    return df_order

if __name__ == "__main__":
    audit_files()
