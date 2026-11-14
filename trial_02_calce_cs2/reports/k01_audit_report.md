# K01 Data Audit Report: CALCE CS2 Cell (`CS2_35`)

**Date:** 2026-09-17  
**Dataset:** CALCE CS2 Prismatic LiCoO2 Cells (`CS2_35` to `CS2_38`)  
**Source:** University of Maryland CALCE Battery Research Group  

---

## 1. Temperature Channel Audit

### Finding
**ZERO temperature channels exist.**

In the Arbin cycler data sheet (`Channel_1-008`), exactly 17 columns are recorded:
1. `Data_Point`
2. `Test_Time(s)`
3. `Date_Time`
4. `Step_Time(s)`
5. `Step_Index`
6. `Cycle_Index`
7. `Current(A)`
8. `Voltage(V)`
9. `Charge_Capacity(Ah)`
10. `Discharge_Capacity(Ah)`
11. `Charge_Energy(Wh)`
12. `Discharge_Energy(Wh)`
13. `dV/dt(V/s)`
14. `Internal_Resistance(Ohm)`
15. `Is_FC_Data`
16. `AC_Impedance(Ohm)`
17. `ACI_Phase_Angle(Deg)`

- The substring match `deg` belongs exclusively to `ACI_Phase_Angle(Deg)` (the electrical phase angle of electrochemical impedance).
- No thermocouple was affixed to CS2 cells during these continuous 1C cycling tests.

### Architectural Implication for Pipeline (K03)
All temperature features used in NASA prognostic pipelines ($T$, $dT/dt$, $\Delta T_{rise}$, rolling mean temperature, rolling std temperature) are uncomputable on CALCE. The feature space must be restricted to **`COMMON_FEATURES`**:
$$\text{Voltage}, \text{Current}, \frac{dV}{dt}, \frac{dI}{dt}, \text{Power} (V \cdot I), t_{elapsed}, \text{Voltage rolling stats (float64)}$$

---

## 2. Multi-File Chronological Ordering Audit

### Finding: The 3 Critical Pipeline Traps

#### Trap A: Filename Date Encoding & Lexicographical Sorting Reversal
Files are named with non-zero-padded date strings: `CS2_35_{M}_{D}_{YY}.xlsx`.
- Standard Python `sorted(os.listdir())` sorts lexicographically:
  `CS2_35_10_15_10.xlsx` (Oct 2010) $\to$ `CS2_35_1_10_11.xlsx` (Jan 2011) $\to \dots \to$ `CS2_35_8_17_10.xlsx` (Aug 2010).
- This completely reverses and scrambles cell history: fresh cycles recorded in August/September 2010 are placed **after** heavily aged cycles from 2011.

#### Trap B: `Cycle_Index` Resets to 1 in Every File
Inside each Arbin `.xlsx` file, the cycler hardware restarts `Cycle_Index` at `1`:
- File 0: cycles 1–1
- File 1: cycles 1–1
- File 2: cycles 1–1
- File 3: cycles 1–50
- File 4: cycles 1–45
- ...
- File 23: cycles 1–50

**Rule:** `Cycle_Index` cannot be used directly as a global cycle counter. Global cycle index must be computed cumulatively based on chronological file chaining:
$$\text{global\_cycle} = \text{cycle\_offset} + \text{local\_cycle}$$

#### Trap C: Duplicate Log Files
Files `CS2_35_2_4_11.xlsx` and `CS2_35_2_10_11.xlsx` contain byte-identical data (`df1.equals(df2) == True`), spanning the exact same test interval `2011-01-31 10:51:35` to `2011-02-03 18:10:51` (50 cycles).
- **Rule:** Deduplication based on `(Date_Time.first(), Date_Time.last(), row_count)` is required to prevent repeating the final 50 cycles.

---

## 3. Verified Chronological Timeline for `CS2_35`

Sorted strictly by `Date_Time` with duplicate removal:

| Seq | Filename | Start Date-Time | End Date-Time | Cycles | Cumulative Cycles |
|:---|:---|:---|:---|:---:|:---:|
| 1 | `CS2_35_8_17_10.xlsx` | 2010-08-16 13:44:57 | 2010-08-16 17:24:02 | 1 | 1 |
| 2 | `CS2_35_8_18_10.xlsx` | 2010-08-17 14:30:57 | 2010-08-17 18:06:57 | 1 | 2 |
| 3 | `CS2_35_8_19_10.xlsx` | 2010-08-18 10:59:23 | 2010-08-18 14:35:03 | 1 | 3 |
| 4 | `CS2_35_8_30_10.xlsx` | 2010-08-19 14:21:41 | 2010-08-26 17:26:06 | 50 | 53 |
| 5 | `CS2_35_9_7_10.xlsx` | 2010-08-31 13:30:15 | 2010-09-06 16:52:06 | 45 | 98 |
| 6 | `CS2_35_9_8_10.xlsx` | 2010-09-07 10:44:17 | 2010-09-08 09:09:17 | 7 | 105 |
| 7 | `CS2_35_9_21_10.xlsx` | 2010-09-14 13:57:43 | 2010-09-21 09:08:16 | 50 | 155 |
| 8 | `CS2_35_9_30_10.xlsx` | 2010-09-21 15:48:03 | 2010-09-28 07:30:12 | 50 | 205 |
| 9 | `CS2_35_10_15_10.xlsx` | 2010-10-08 14:29:45 | 2010-10-15 10:03:54 | 50 | 255 |
| 10 | `CS2_35_10_22_10.xlsx` | 2010-10-15 17:12:56 | 2010-10-22 10:41:09 | 50 | 305 |
| 11 | `CS2_35_10_29_10.xlsx` | 2010-10-22 13:23:55 | 2010-10-29 04:44:14 | 50 | 355 |
| 12 | `CS2_35_11_01_10.xlsx` | 2010-10-29 09:58:03 | 2010-10-30 17:54:37 | 10 | 365 |
| 13 | `CS2_35_11_08_10.xlsx` | 2010-11-01 14:23:58 | 2010-11-08 05:55:51 | 50 | 415 |
| 14 | `CS2_35_11_23_10.xlsx` | 2010-11-16 11:45:27 | 2010-11-23 02:33:49 | 50 | 465 |
| 15 | `CS2_35_11_24_10.xlsx` | 2010-11-23 12:25:25 | 2010-11-24 15:05:43 | 9 | 474 |
| 16 | `CS2_35_12_06_10.xlsx` | 2010-11-29 10:34:08 | 2010-12-05 20:21:50 | 50 | 524 |
| 17 | `CS2_35_12_13_10.xlsx` | 2010-12-06 12:33:56 | 2010-12-12 21:35:39 | 50 | 574 |
| 18 | `CS2_35_12_20_10.xlsx` | 2010-12-13 12:36:53 | 2010-12-19 17:34:39 | 50 | 624 |
| 19 | `CS2_35_12_23_10.xlsx` | 2010-12-20 12:10:10 | 2010-12-23 12:37:28 | 25 | 649 |
| 20 | `CS2_35_1_10_11.xlsx` | 2011-01-03 10:38:25 | 2011-01-09 02:13:19 | 50 | 699 |
| 21 | `CS2_35_1_18_11.xlsx` | 2011-01-10 11:47:49 | 2011-01-15 17:17:16 | 50 | 749 |
| 22 | `CS2_35_1_24_11.xlsx` | 2011-01-18 13:17:53 | 2011-02-23 12:30:50 | 50 | 799 |
| 23 | `CS2_35_1_28_11.xlsx` | 2011-01-24 10:54:44 | 2011-01-27 18:41:37 | 37 | 836 |
| 24 | `CS2_35_2_10_11.xlsx` *(deduped)* | 2011-01-31 10:51:35 | 2011-02-03 18:10:51 | 50 | **886** |

**Total Verified Lifespan:** **886 unique continuous cycles** across 6 months of aging.
