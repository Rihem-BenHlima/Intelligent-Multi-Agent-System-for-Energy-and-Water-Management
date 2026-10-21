"""
ETL pipeline for water forecasting data preparation.
Cleans, imputes, and validates hourly water consumption records.
"""

import contextlib
import io
import os
import sys
from pathlib import Path
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Import Step 2 and Step 3 from water_forecasting/etl
_env = os.getenv("ML_MODELS_DIR")
_ROOT = Path(_env) if _env else Path(__file__).resolve().parent.parent.parent
_WATER_ETL_PATH = str(_ROOT / "water_forecasting" / "etl")
if _WATER_ETL_PATH not in sys.path:
    sys.path.insert(0, _WATER_ETL_PATH)

sys.modules.pop("config", None)
sys.modules.pop("step2_detect_impute", None)
sys.modules.pop("step3_smooth_validate", None)
import step2_detect_impute as _real_step2
import step3_smooth_validate as _real_step3

# Step 1 configuration
RATE_CAP_L_PER_H     = 10_000   # hardware spike cap (L/h)
ZERO_RUN_GAP_HOURS   = 96       # consecutive zeros threshold for missing data (hours)
MAX_SYNTHETIC_FRAC   = 0.05     # soft warning threshold for missing ratio
HOUSEHOLD_ID         = "Panther_office_Larry"


# ── Step 1 ────────────────────────────────────────────────────────────────────
def step1_clean(df_mysql: pd.DataFrame) -> pd.DataFrame:
    """Clean raw water records, apply rate capping, and flag long zero-run gaps."""
    df = df_mysql.copy()
    df["datetime"]     = pd.to_datetime(df["date"], unit="ms")
    df["consumption_l"] = df["total_consumption"].astype(float)
    df["household_id"] = HOUSEHOLD_ID
    df = df.sort_values("datetime").drop_duplicates(subset=["datetime"]).reset_index(drop=True)

    # Complete the timeline
    full_idx = pd.date_range(df["datetime"].min(), df["datetime"].max(), freq="h")
    missing  = len(full_idx) - len(df)
    if missing != 0:
        df = (df.set_index("datetime").reindex(full_idx)
                .rename_axis("datetime").reset_index())
        df["household_id"] = df["household_id"].fillna(HOUSEHOLD_ID)

    df["consumption_l_prespike"] = df["consumption_l"].copy()

    # Hardware spike cap
    spike_mask = df["consumption_l"] > RATE_CAP_L_PER_H
    df.loc[spike_mask, "consumption_l"] = np.nan

    # Zero-run gap flagging for consecutive missing data
    z       = (df["consumption_l"] == 0).astype(int)
    run_id  = (z != z.shift()).cumsum()
    df["_zero_run_len"] = df.groupby(run_id)["consumption_l"].transform(
        lambda s: len(s) if (s == 0).all() else 0
    )
    long_zero_mask = (df["consumption_l"] == 0) & (df["_zero_run_len"] >= ZERO_RUN_GAP_HOURS)
    df.loc[long_zero_mask, "consumption_l"] = np.nan
    df = df.drop(columns=["_zero_run_len"])

    # Context columns
    df["hour_of_day"] = df["datetime"].dt.hour
    df["dayofweek"]   = df["datetime"].dt.dayofweek

    total_nan = df["consumption_l"].isna().sum()
    if len(df) > 0 and total_nan / len(df) > MAX_SYNTHETIC_FRAC:
        import logging
        logging.getLogger(__name__).warning(
            "Water ETL Step 1: %.1f%% NaN after cleaning (threshold %.0f%%). "
            "Proceeding with imputation.", total_nan / len(df) * 100,
            MAX_SYNTHETIC_FRAC * 100
        )

    return df[["household_id", "datetime", "consumption_l",
               "consumption_l_prespike", "hour_of_day", "dayofweek"]]


def step2_impute(df: pd.DataFrame) -> pd.DataFrame:
    """Run tiered imputation delegating to water_forecasting/etl/step2_detect_impute.py."""
    return _real_step2.run(df)


def step3_validate(df: pd.DataFrame) -> pd.DataFrame:
    """Run validation delegating to water_forecasting/etl/step3_smooth_validate.py."""
    out, _ = _real_step3.run(df, sigma=None)
    return out


def run(df_mysql: pd.DataFrame) -> pd.DataFrame:
    """Run full water forecasting ETL pipeline (clean, impute, and validate)."""
    s1 = step1_clean(df_mysql)
    with contextlib.redirect_stdout(io.StringIO()):
        s2 = step2_impute(s1)
        s3 = step3_validate(s2)
    return s3
