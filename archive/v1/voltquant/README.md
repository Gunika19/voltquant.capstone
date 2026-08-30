# ⚙️ VoltQuant — Dataset Standardization Guide

---

# 📌 Purpose

This document explains how to **standardize battery datasets** in VoltQuant.

⚠️ This is the **most important step** in Phase 1.
All future ML models depend on clean, consistent data.

---

# 🎯 Goal

Convert all datasets into a **common schema**:

```text
voltage | temperature | current | cycle | time
```

---

# 🚨 Problem

Battery datasets (NASA, EV datasets, etc.) are:

* inconsistent column names
* different units
* missing fields
* different formats

---

# ✅ Solution

We use a **mapping-based standardization system**.

Instead of editing raw data manually:

```text
Raw Dataset → Mapping → Standardized Dataset
```

---

# 🧠 Key Concept

We define how each dataset maps to the standard format.

Example:

```python
{
    "Voltage_measured": "voltage",
    "Temperature_measured": "temperature",
    "Current_measured": "current",
    "Cycle_Index": "cycle",
    "Time": "time"
}
```

---

# ⚙️ Implementation Steps

---

## 🔹 Step 1 — Create standardization module

Create file:

```bash
src/voltquant/nodes/standardize.py
```

---

### Paste this code:

```python
import pandas as pd

STANDARD_COLUMNS = ["voltage", "temperature", "current", "cycle", "time"]

def standardize_dataset(df: pd.DataFrame, column_mapping: dict):
    """
    Convert dataset into standard schema using mapping
    """

    # Rename columns
    df = df.rename(columns=column_mapping)

    # Keep only required columns
    df = df[[col for col in STANDARD_COLUMNS if col in df.columns]]

    # Add missing columns
    for col in STANDARD_COLUMNS:
        if col not in df.columns:
            df[col] = None

    # Reorder columns
    df = df[STANDARD_COLUMNS]

    return df
```

---

## 🔹 Step 2 — Create dataset-specific mapping

Each dataset needs its own mapping.

Example (NASA dataset):

```python
mapping = {
    "Voltage_measured": "voltage",
    "Temperature_measured": "temperature",
    "Current_measured": "current",
    "Cycle_Index": "cycle",
    "Time": "time"
}
```

---

## 🔹 Step 3 — Create standardization pipeline

### nodes.py

```python
from voltquant.nodes.standardize import standardize_dataset

def standardize_data(df):
    mapping = {
        "Voltage_measured": "voltage",
        "Temperature_measured": "temperature",
        "Current_measured": "current",
        "Cycle_Index": "cycle",
        "Time": "time"
    }

    return standardize_dataset(df, mapping)
```

---

### pipeline.py

```python
from kedro.pipeline import Pipeline, node
from .nodes import standardize_data

def create_pipeline(**kwargs):
    return Pipeline([
        node(standardize_data, "raw_data", "standardized_data"),
    ])
```

---

## 🔹 Step 4 — Register pipeline

Update:

```python
src/voltquant/pipeline_registry.py
```

```python
from voltquant.pipelines.preprocessing.pipeline import create_pipeline as preprocessing_pipeline
from voltquant.pipelines.standardization.pipeline import create_pipeline as standardization_pipeline

def register_pipelines():
    return {
        "__default__": preprocessing_pipeline(),
        "preprocessing": preprocessing_pipeline(),
        "standardization": standardization_pipeline(),
    }
```

---

## 🔹 Step 5 — Update catalog

Edit:

```yaml
conf/base/catalog.yml
```

Add:

```yaml
raw_data:
  type: pandas.CSVDataset
  filepath: data/01_raw/raw.csv

standardized_data:
  type: pandas.CSVDataset
  filepath: data/02_intermediate/standardized.csv
```

---

## 🔹 Step 6 — Run pipeline

```bash
kedro run --pipeline standardization
```

---

# 👥 Team Responsibilities

Each team member must:

1. Select one dataset
2. Analyze:

   * column names
   * units
   * missing values
3. Create a mapping dictionary
4. Run standardization
5. Verify output

---

# ⚠️ Rules (STRICT)

* ❌ Do NOT edit raw datasets manually

* ❌ Do NOT push large datasets to GitHub

* ❌ Do NOT change schema format

* ✅ Always use mapping

* ✅ Keep transformations reproducible

* ✅ Write clean code

---

# 🧪 Output

Final dataset will be stored in:

```text
data/02_intermediate/standardized.csv
```

---

# 🚀 Why this matters

If this step is weak:

❌ Models will fail
❌ Results will be inconsistent
❌ Project becomes unscalable

If done properly:

✅ Clean ML pipeline
✅ Easy dataset expansion
✅ Research-grade system

---

# 🔥 Next Steps

After standardization:

* Dataset merging
* Unit normalization
* Feature engineering (PCA)
* ML models

---

# 🧠 Pro Tip

Do NOT rush this step.

> Good data = good models
> Bad data = useless system

---

# 📌 Final Note

This is not just preprocessing.

This is the **foundation of the entire VoltQuant system**.
