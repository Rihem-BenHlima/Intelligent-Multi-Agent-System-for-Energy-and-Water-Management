# Water anomaly dataset — PUBLIC (Building Data Genome Project 2)

The water data is **public**: BDG2, site `Panther`, office building `Larry`, hourly 2016-2017.
Full description, links, citation and rebuild steps:
[`water_forecasting/bdg2_office_water/README.md`](../../water_forecasting/bdg2_office_water/README.md)

- GitHub: <https://github.com/buds-lab/building-data-genome-project-2>

```
water_anomaly_detection/dataset/water_consumption_office.csv
```

This pipeline uses the **original, un-shifted 2016-2017** series. `step1_etl_water.py`
auto-detects either litres (`consumption_l`) or m³ (`consumption_m3`, converted ×1000)
columns, so the file from the water_forecasting folder works as-is.
