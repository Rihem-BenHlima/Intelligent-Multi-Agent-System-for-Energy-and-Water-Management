# Energy dataset — ENTERPRISE 

## What the dataset is

- Source: electricity meter readings of a single site, single channel (meter serial
  number configured in `energy_forecasting/etl/config.py` → `EXTRACTION["sn"]`).
- Granularity: sub-minute **consumption deltas** (kWh per reading), later resampled to hourly.
- Length: several months of continuous history is recommended (more is better for the weekly/seasonal features).

`raw_data.csv` / `raw_data_extended.csv` format:

| column | example | meaning |
|---|---|---|
| `date` | `yyyy-mm-dd hh:mmm:ss` | reading timestamp |
| `sn` | `YOUR_METER_SERIAL_NUMBER` | meter serial number |
| `consumption_diff` | `x.xxxxxx` | kWh consumed since previous reading |

## How to provide your own data

1. **From a database:** fill the `DB_*` variables in the root `.env` and run
   `python energy_forecasting/etl/step0_extract_arrange.py`.
2. **From a file:** drop a CSV with the 3 columns above into `raw/raw_data.csv`.
3. Any open smart-meter dataset converted to this format works for testing.

The same raw file is also expected by `energy_anomaly_detection/dataset/raw_data.csv`.