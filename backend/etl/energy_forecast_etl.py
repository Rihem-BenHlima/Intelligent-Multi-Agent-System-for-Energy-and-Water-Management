"""
ETL pipeline for energy forecasting data preparation.
Cleans, imputes, and validates hourly energy consumption records.
"""

import contextlib
import io
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Import Step 2 and Step 3 from energy_forecasting/etl
_env = os.getenv("ML_MODELS_DIR")
_ROOT = Path(_env) if _env else Path(__file__).resolve().parent.parent.parent
_ENERGY_ETL_PATH = str(_ROOT / "energy_forecasting" / "etl")
if _ENERGY_ETL_PATH not in sys.path:
    sys.path.insert(0, _ENERGY_ETL_PATH)

for _m in ("config", "step2_detect_impute", "step3_smooth_validate"):
    sys.modules.pop(_m, None)
import step2_detect_impute as _real_step2      
import step3_smooth_validate as _real_step3    

# Step 1 configuration
WEEKEND = [5, 6]
SEASONS = {
    "summer": [6, 7, 8],
    "winter": [12, 1, 2],
    "spring": [3, 4, 5],
    "autumn": [9, 10, 11],
}
SPIKE_Q = 0.9995


def _get_season(month: int) -> str:
    for name, months in SEASONS.items():
        if month in months:
            return name
    return "autumn"


def step1_clean(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Clean and structure raw hourly records into standard feature columns."""
    df = df_raw.copy()
    df["datetime"] = pd.to_datetime(df["date"], unit="ms")
    df = df.rename(columns={"total_consumption": "energy_kwh"})
    # MODIFY: Set your meter/device serial number here
    df["sn"] = "YOUR_METER_SERIAL"
    df = df.sort_values("datetime").drop_duplicates(subset=["datetime"]).reset_index(drop=True)

    # Seasonal spike cap (same as training step 1)
    df["month"] = df["datetime"].dt.month
    df["season"] = df["month"].apply(_get_season)
    global_cap = df["energy_kwh"].quantile(SPIKE_Q)
    caps = {
        s: max(df[df["season"] == s]["energy_kwh"].quantile(SPIKE_Q), global_cap * 0.8)
        for s in df["season"].unique()
    }
    df["cap"] = df["season"].map(caps)
    df.loc[df["energy_kwh"] > df["cap"], "energy_kwh"] = np.nan

    # Complete the timeline (no gaps) so step 2's gap classification is correct
    full_idx = pd.date_range(df["datetime"].min(), df["datetime"].max(), freq="h")
    df = df.set_index("datetime").reindex(full_idx).rename_axis("datetime").reset_index()
    df["sn"] = df["sn"].ffill().bfill()

    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["season"] = df["month"].apply(_get_season)
    df["is_weekend"] = df["dayofweek"].isin(WEEKEND).astype(int)

    return df[["sn", "datetime", "energy_kwh", "hour", "dayofweek",
               "month", "season", "is_weekend"]]


def run(df_mysql: pd.DataFrame) -> pd.DataFrame:
    """Run full energy forecasting ETL pipeline (clean, impute, smooth, and validate)."""
    s1 = step1_clean(df_mysql)
    with contextlib.redirect_stdout(io.StringIO()):
        s2 = _real_step2.run(s1)
        s3 = _real_step3.run(s2)
    return s3
