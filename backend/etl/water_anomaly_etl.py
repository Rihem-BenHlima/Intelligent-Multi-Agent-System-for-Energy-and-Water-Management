"""
ETL pipeline for water anomaly detection data preparation.
Prepares raw hourly water consumption records while preserving real consumption anomalies.
"""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Configuration
SPIKE_Q     = 0.9995   # global hardware-spike percentile
IQR_K       = 3.5
HOUSEHOLD_ID = "Panther_office_Larry"


def load_hourly(df_mysql: pd.DataFrame) -> pd.DataFrame:
    """Structure raw hourly records, apply initial capping, and store original values."""
    df = df_mysql.copy()
    df["datetime"]     = pd.to_datetime(df["date"], unit="ms")
    df["consumption_l"] = df["total_consumption"].astype(float)
    df["sn"]           = HOUSEHOLD_ID
    df = df.sort_values("datetime").drop_duplicates(subset=["datetime"]).reset_index(drop=True)

    # Complete the timeline
    full_idx = pd.date_range(df["datetime"].min(), df["datetime"].max(), freq="h")
    df = (df.set_index("datetime").reindex(full_idx)
            .rename_axis("datetime").reset_index())
    df["sn"] = df["sn"].fillna(HOUSEHOLD_ID)

    # Save original before capping
    df["consumption_l_original"] = df["consumption_l"].copy()

    # Hardware-spike cap
    global_cap = df["consumption_l"].quantile(SPIKE_Q)
    spikes     = df["consumption_l"] > global_cap
    df.loc[spikes, "consumption_l"] = np.nan

    # Context columns
    df["hour_of_day"] = df["datetime"].dt.hour
    df["dayofweek"]   = df["datetime"].dt.dayofweek

    return df


def prepare_for_detection(df: pd.DataFrame) -> pd.DataFrame:
    """Restore spike values and fill coverage gaps while preserving anomaly patterns."""
    df = df.copy()

    # Restore spikes from original column
    restore_mask = df["consumption_l"].isna() & df["consumption_l_original"].notna()
    df.loc[restore_mask, "consumption_l"] = df.loc[restore_mask, "consumption_l_original"]

    # Fill true coverage gaps
    df["consumption_l"] = df["consumption_l"].ffill().bfill().clip(lower=0)

    return df


def run(df_mysql: pd.DataFrame) -> pd.DataFrame:
    """Run water anomaly detection ETL pipeline."""
    base   = load_hourly(df_mysql)
    df_raw = prepare_for_detection(base.copy())

    cols = ["sn", "datetime", "consumption_l", "hour_of_day", "dayofweek"]
    return df_raw[cols]
