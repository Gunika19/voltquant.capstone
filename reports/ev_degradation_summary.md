# ev_battery_degradation_v1.csv: descriptive summary

Documentation for the paper, not a pipeline stage -- this source is not cleaned into a feature table because it cannot support a supervised claim (see the note at the end).

## Shape

10000 rows x 13 columns.

## Columns

| column | description |
|---|---|
| `Vehicle_ID` | Opaque per-vehicle identifier (hex string). |
| `Car_Model` | Vehicle make/model (5 distinct values). |
| `Battery_Type` | Cell chemistry: NMC or LFP. |
| `Battery_Capacity_kWh` | Nameplate pack capacity, kWh. |
| `Vehicle_Age_Months` | Vehicle age, months. |
| `Total_Charging_Cycles` | Cumulative charge cycles to date. |
| `Avg_Temperature_C` | Average operating temperature, degrees C. |
| `Fast_Charge_Ratio` | Fraction of charges that were fast-charges (0-1). |
| `Avg_Discharge_Rate_C` | Average discharge rate, C-rate units. |
| `Driving_Style` | Categorical: Aggressive, Moderate, or Conservative. |
| `Internal_Resistance_Ohm` | Internal resistance, ohms. |
| `SoH_Percent` | State of health, percent (100 = new). |
| `Battery_Status` | Categorical label: Healthy or Replace Required. |

## Battery_Status counts

| status | count | pct |
|---|---|---|
| Healthy | 9996 | 99.96% |
| Replace Required | 4 | 0.04% |

## SoH_Percent distribution by Battery_Status

| status | count | mean | std | min | 25% | 50% | 75% | max |
|---|---|---|---|---|---|---|---|---|
| Healthy | 9996 | 95.81 | 3.23 | 80.17 | 94.03 | 96.50 | 98.31 | 100.00 |
| Replace Required | 4 | 77.08 | 2.32 | 73.74 | 76.46 | 77.81 | 78.44 | 78.96 |

## Note

With 4 positive(s) (`Replace Required`) in 10000 rows (0.04%), this source is documented context for the paper, not training data -- 4 examples cannot support any supervised claim, regardless of how separable they look in `SoH_Percent` (see the distribution above).
