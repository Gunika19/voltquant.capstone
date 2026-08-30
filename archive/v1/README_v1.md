# Voltquant - 19/04 Push (Structure and Instructions)

### Hybrid AI + Hardware System for Battery Intelligence

---

# 📌 1. What is VoltQuant?

VoltQuant is a **modular machine learning + hardware system** designed to:

* Predict battery performance
* Detect thermal runaway risks
* Analyze EV battery behavior

It combines:

* 📊 Classical Machine Learning
* ⚛️ Hybrid Quantum ML (HQML) *(future work)*
* 🔌 FPGA Acceleration (PYNQ-Z2) *(future work)*
* 📡 Edge Systems (ESP32) *(future work)*

---

# 🎯 2. Current Status (IMPORTANT)

## ✅ What is DONE right now

We have successfully built:

* ✔️ Project structure using **Kedro**
* ✔️ Virtual environment setup
* ✔️ Data pipeline system
* ✔️ Preprocessing pipeline:

  * Merge datasets
  * Clean data
  * Placeholder for augmentation
* ✔️ GitHub-ready repository
* ✔️ Modular architecture for scaling

---

## ❌ What is NOT done yet

* GAN-based augmentation
* PCA / feature engineering
* ML models
* HQML
* Hardware deployment

👉 **Right now, we are in the DATA ENGINEERING PHASE**

---

# 🧠 3. What is Kedro (for complete beginners)

Kedro is a framework that forces you to structure ML projects properly.

Instead of messy scripts:

| Concept  | Meaning                              |
| -------- | ------------------------------------ |
| Node     | A single function (e.g., clean data) |
| Pipeline | A sequence of nodes                  |
| Catalog  | Where data is stored                 |
| Project  | Entire system                        |

---

## 🔄 How Kedro works (our project)

```text
Raw Data → Node → Node → Node → Output Data
```

In VoltQuant:

```text
data1 + data2
      ↓
merge_datasets
      ↓
clean_data
      ↓
apply_gan (placeholder)
      ↓
augmented_data
```

---

# 📁 4. Project Structure (VERY IMPORTANT)

```text
voltquant/
│
├── conf/                  # Configuration (Kedro settings)
│   └── base/
│       └── catalog.yml    # Defines datasets
│
├── data/                  # Data storage (DO NOT PUSH LARGE FILES)
│   ├── 01_raw/            # Raw datasets (NASA etc.)
│   ├── 02_intermediate/   # Cleaned/merged data
│   ├── 03_primary/
│   ├── 04_feature/
│   ├── 05_model_input/
│   ├── 06_models/
│   ├── 07_model_output/
│   └── 08_reporting/
│
├── src/voltquant/
│
│   ├── pipelines/         # Main pipelines
│   │   ├── preprocessing/ # CURRENT WORKING PIPELINE
│   │   ├── dimensionality/
│   │   ├── classical_models/
│   │   ├── hqml/
│   │   └── hardware/
│
│   ├── agents/            # Future: AI agent orchestration
│   ├── nodes/             # Reusable functions
│   └── utils/             # Helper functions
│
├── notebooks/             # Experiments (not production)
├── tests/                 # Testing
├── requirements.txt       # Dependencies
└── pyproject.toml
```

---

# ⚙️ 5. Setup Instructions (for team members)

## Step 1 — Clone repo

```bash
git clone <repo-url>
cd voltquant
```

---

## Step 2 — Create environment

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

---

## Step 4 — Run pipeline

```bash
kedro run
```

---

# 🔬 6. Current Pipeline Explained (VERY IMPORTANT)

## File: `nodes.py`

Contains actual logic:

* `merge_datasets` → combines multiple datasets
* `clean_data` → removes missing values
* `apply_gan` → placeholder (will be upgraded)

---

## File: `pipeline.py`

Defines workflow:

```text
merge → clean → augment
```

---

## File: `catalog.yml`

Defines:

* where data comes from
* where outputs are saved

Example:

```yaml
data1:
  type: pandas.CSVDataset
  filepath: data/01_raw/data1.csv
```

---

# 📊 7. TEAM TASK — Dataset Standardization

## 🚨 Problem

Battery datasets (NASA etc.) are:

* inconsistent column names
* different units
* missing fields
* different formats

---

## 🎯 Goal

Create a **standard dataset format**:

```text
voltage | temperature | current | cycle | time
```

---

## 👥 Each team member should:

1. Pick ONE dataset
2. Analyze:

   * column names
   * units
   * missing values
3. Convert it into standard format
4. Save in:

```text
data/01_raw/
```

---

## ⚠️ Rules

* ❌ Do NOT modify original dataset directly
* ❌ Do NOT push large datasets to GitHub
* ✅ Keep transformations reproducible
* ✅ Write code, not manual edits

---

# 🤖 8. Agents (Future Concept)

We are designing an **agent-based system**:

| Agent          | Role                         |
| -------------- | ---------------------------- |
| Data Agent     | preprocessing + augmentation |
| Training Agent | ML models                    |
| HQML Agent     | quantum models               |
| Hardware Agent | FPGA + ESP32                 |

👉 Not implemented yet, but structure is ready

---

# 🔌 9. Future Roadmap

## Phase 1 (NOW)

* Dataset merging
* Cleaning
* Standardization

## Phase 2

* GAN augmentation (CTGAN / TimeGAN)
* PCA + feature selection

## Phase 3

* Classical ML models (baseline)

## Phase 4

* HQML (core innovation)

## Phase 5

* FPGA + ESP32 deployment

---

# 🧠 10. How to Use This Project with AI Tools

You can paste this README into:

* ChatGPT
* Claude
* Copilot

And ask:

* “Explain preprocessing pipeline”
* “Help implement GAN node”
* “Add PCA pipeline”
* “Design HQML architecture”

👉 This README is designed to be **AI-readable context**

---

# 👥 11. Team Workflow

* Create feature branches:

```bash
git checkout -b feature/preprocessing
```

* Commit changes:

```bash
git commit -m "added dataset cleaning logic"
```

* Push:

```bash
git push origin feature/preprocessing
```

* Create Pull Request

---

# ⚠️ 12. Important Rules

* DO NOT work directly on `main`
* DO NOT push raw datasets
* KEEP functions modular

---
