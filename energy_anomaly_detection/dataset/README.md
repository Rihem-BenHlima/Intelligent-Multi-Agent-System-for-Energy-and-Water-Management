# Energy anomaly dataset — ENTERPRISE 

Place the **enterprise energy meter file** here as `raw_data.csv`.
It is the same data described in
[`energy_forecasting/etl/data/README.md`](../../energy_forecasting/etl/data/README.md).

Expected path (see `energy_anomaly_detection/config.py` → `RAW_DATA_CSV`):

```
energy_anomaly_detection/dataset/raw_data.csv
```

Columns: `date` (timestamp), `sn` (meter serial), `consumption_diff` (kWh since previous reading).
Use your own meter export in this format to run `step1_etl.py` → `step4_export_detection_config.py`.
