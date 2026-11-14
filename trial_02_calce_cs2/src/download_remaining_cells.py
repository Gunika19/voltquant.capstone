"""Download and extract remaining CALCE CS2 cells: CS2_36, CS2_37, CS2_38."""

import os
import urllib.request
import zipfile
from pathlib import Path

RAW_DIR = Path("data/01_raw/calce")
RAW_DIR.mkdir(parents=True, exist_ok=True)

CELLS = ["CS2_36", "CS2_37", "CS2_38"]

for cell in CELLS:
    zip_path = RAW_DIR / f"{cell}.zip"
    extract_dir = RAW_DIR / cell
    url = f"https://web.calce.umd.edu/batteries/data/{cell}.zip"
    
    if not zip_path.exists():
        print(f"Downloading {cell} from {url}...")
        urllib.request.urlretrieve(url, zip_path)
        print(f"Downloaded {zip_path.name} ({zip_path.stat().st_size:,} bytes).")
    else:
        print(f"{zip_path.name} already exists ({zip_path.stat().st_size:,} bytes).")
        
    if not extract_dir.exists():
        print(f"Extracting to {extract_dir}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"Extraction complete for {cell}.")
    else:
        print(f"{extract_dir} already extracted.")

print("All remaining CALCE CS2 cells ready.")
