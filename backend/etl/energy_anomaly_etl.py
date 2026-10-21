"""
ETL pipeline for energy anomaly detection data preparation.
Prepares raw hourly consumption records while preserving real consumption anomalies.
"""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Domain constants
SPIKE_Q    = 0.9995
COV_THRESH = 0.35
MIN_READ   = 20
WORK_START = 8
WORK_END   = 18
WEEKEND    = [5, 6]
SEASONS    = {
    "summer": [6, 7, 8],
    "winter": [12, 1, 2],
    "spring": [3, 4, 5],
    "autumn": [9, 10, 11],
}


def _get_season(month: int) -> str:
    for name, months in SEASONS.items():
        if month in months:
            return name
    return "autumn"


def load_and_aggregate(df_mysql: pd.DataFrame) -> pd.DataFrame:
    """Structure raw hourly records, remove extreme sensor spikes, and add calendar features."""
    df = df_mysql.copy()
    df["datetime"]   = pd.to_datetime(df["date"], unit="ms")
    df["energy_kwh"] = df["total_consumption"].astype(float)
    # MODIFY: Set your meter/device serial number here
    df["sn"]         = "YOUR_METER_SERIAL"
    df = df.sort_values("datetime").drop_duplicates(subset=["datetime"]).reset_index(drop=True)

    # Seasonal spike removal (hardware artefacts — NOT consumption anomalies)
    df["month"]  = df["datetime"].dt.month
    df["season"] = df["month"].apply(_get_season)
    global_cap   = df["energy_kwh"].quantile(SPIKE_Q)
    caps = {
        s: max(df[df["season"] == s]["energy_kwh"].quantile(SPIKE_Q), global_cap * 0.8)
        for s in df["season"].unique()
    }
    df["cap"]   = df["season"].map(caps)
    spikes      = df["energy_kwh"] > df["cap"]
    df.loc[spikes, "energy_kwh"] = np.nan

    # Complete the timeline
    full_idx = pd.date_range(df["datetime"].min(), df["datetime"].max(), freq="h")
    df = df.set_index("datetime").reindex(full_idx).rename_axis("datetime").reset_index()
    df["sn"] = df["sn"].ffill()

    # Context columns — aligned to energy_forecast_etl column names
    df["hour"]       = df["datetime"].dt.hour          # was hour_of_day
    df["dayofweek"]  = df["datetime"].dt.dayofweek
    df["month"]      = df["datetime"].dt.month
    df["season"]     = df["month"].apply(_get_season)
    df["is_weekend"] = df["dayofweek"].isin(WEEKEND).astype(int)
    df["is_work_hours"] = (                            # was is_work
        (~df["dayofweek"].isin(WEEKEND)) &
        (df["hour"] >= WORK_START) &
        (df["hour"] <= WORK_END)
    ).astype(int)

    return df


def prepare_for_detection(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values while preserving anomaly patterns."""
    df = df.copy()
    df["energy_kwh"] = df["energy_kwh"].ffill().bfill().clip(lower=0)
    return df


def run(df_mysql: pd.DataFrame) -> pd.DataFrame:
    """Run energy anomaly detection ETL pipeline."""
    base     = load_and_aggregate(df_mysql)
    df_raw   = prepare_for_detection(base.copy())

    cols = ["sn", "datetime", "energy_kwh", "hour", "dayofweek",
            "month", "season", "is_weekend", "is_work_hours"]
    return df_raw[cols]
