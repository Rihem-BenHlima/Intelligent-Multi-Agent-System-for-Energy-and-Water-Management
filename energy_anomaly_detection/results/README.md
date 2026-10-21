# Energy anomaly results — generated outputs

## What ends up here after detection

Both files are written by `step3_detect.py`:

| File | Content |
|---|---|
| `anomaly_results.csv` | One row per hour of the test period: `timestamp`, `actual_kwh`, `reconstructed_kwh`, `residual_kwh`, reconstruction error (raw + smoothed), the two thresholds, and the point / smooth / final anomaly flags |
| `anomaly_events.csv` | The detected anomaly events: `start`, `end`, `duration_hours`, `max_mse`, `margin`, `signal` (point / smooth / both), `sustained` |