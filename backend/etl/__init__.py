"""
backend/etl/__init__.py
========================
Four separate ETL pipelines — each adapted for the Data Agent.
Data arrives as a MySQL DataFrame; no file I/O inside any ETL.

Usage
-----
from etl.energy_forecast_etl import run as energy_forecast_etl
from etl.energy_anomaly_etl   import run as energy_anomaly_etl
from etl.water_forecast_etl   import run as water_forecast_etl
from etl.water_anomaly_etl    import run as water_anomaly_etl

cleaned_df = energy_forecast_etl(df_from_mysql)
"""
from etl.energy_forecast_etl import run as energy_forecast_etl
from etl.energy_anomaly_etl  import run as energy_anomaly_etl
from etl.water_forecast_etl  import run as water_forecast_etl
from etl.water_anomaly_etl   import run as water_anomaly_etl

__all__ = [
    "energy_forecast_etl",
    "energy_anomaly_etl",
    "water_forecast_etl",
    "water_anomaly_etl",
]
