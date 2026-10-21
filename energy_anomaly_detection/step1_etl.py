"""
step1_etl.py — ETL Pipeline
============================
Input : dataset/raw_data.csv
Output: etl_output/etl_clean.csv   (fully cleaned hourly data — for training)
        etl_output/etl_raw.csv     (minimal processing, anomalies preserved — for detection)

Two-path design:
  CLEAN path → autoencoder learns normal patterns only
  RAW   path → detection runs on data that still contains real anomalies

Run:  python step1_etl.py
"""

import sys, os, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from config import RAW_DATA_CSV, CLEAN_CSV, RAW_CSV as RAW_OUT, ETL_OUTPUT_DIR

# ── CONFIG ────────────────────────────────────────────────────
RAW_CSV    = RAW_DATA_CSV
os.makedirs(ETL_OUTPUT_DIR, exist_ok=True)

SPIKE_Q    = 0.9995
COV_THRESH = 0.35
MIN_READ   = 20
IQR_K      = 3.5
ZERO_H     = 4
WORK_START = 8
WORK_END   = 18
WEEKEND    = [5, 6]
SEASONS    = {"summer":[6,7,8],"winter":[12,1,2],
              "spring":[3,4,5],"autumn":[9,10,11]}

def get_season(month):
    for name, months in SEASONS.items():
        if month in months: return name
    return "autumn"


def load_and_aggregate(path):
    """Shared step: dedup + spike cap + hourly aggregate. Used by both paths."""
    print(f"  Loading: {path}")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    before = len(df)
    df = df.drop_duplicates().sort_values("date").reset_index(drop=True)
    df = df.drop_duplicates(subset=["sn","date"], keep="first")
    print(f"  Raw rows      : {before:,} → {len(df):,} after dedup")
    print(f"  Date range    : {df['date'].min()} → {df['date'].max()}")

    # Sensor spike removal (hardware artefacts only — NOT consumption anomalies)
    df["month"]  = df["date"].dt.month
    df["season"] = df["month"].apply(get_season)
    global_cap   = df["consumption_diff"].quantile(SPIKE_Q)
    caps = {s: max(df[df["season"]==s]["consumption_diff"].quantile(SPIKE_Q),
                   global_cap * 0.8) for s in df["season"].unique()}
    df["cap"] = df["season"].map(caps)
    spikes = df["consumption_diff"] > df["cap"]
    df.loc[spikes, "consumption_diff"] = np.nan
    print(f"  Spikes removed: {spikes.sum():,}")

    # Hourly aggregation
    df["dt"]   = df["date"].diff().dt.total_seconds()
    df["hour"] = df["date"].dt.floor("h")
    hourly = (
        df.groupby("hour")
        .agg(sn=("sn","first"), energy_kwh=("consumption_diff","sum"),
             valid_readings=("consumption_diff","count"),
             median_dt=("dt","median"))
        .reset_index().rename(columns={"hour":"datetime"})
    )
    hourly["median_dt"] = hourly["median_dt"].fillna(22).replace(0, 22)
    hourly["expected"]  = (3600 / hourly["median_dt"]).round(2)
    hourly["coverage"]  = hourly["valid_readings"] / hourly["expected"]

    # Complete the timeline (fill any missing hours)
    full_idx = pd.date_range(hourly["datetime"].min(),
                             hourly["datetime"].max(), freq="h")
    hourly = hourly.set_index("datetime").reindex(full_idx)
    hourly.index.name = "datetime"
    hourly["sn"] = hourly["sn"].ffill()

    # Low-coverage hours → NaN (true missing data, not anomalies)
    bad = ((hourly["coverage"] < COV_THRESH) |
           (hourly["valid_readings"] < MIN_READ)).fillna(True)
    hourly.loc[bad, "energy_kwh"] = np.nan
    hourly = hourly.reset_index()

    # Context columns
    hourly["hour_of_day"] = hourly["datetime"].dt.hour
    hourly["dayofweek"]   = hourly["datetime"].dt.dayofweek
    hourly["month"]       = hourly["datetime"].dt.month
    hourly["season"]      = hourly["month"].apply(get_season)
    hourly["is_weekend"]  = hourly["dayofweek"].isin(WEEKEND).astype(int)
    hourly["is_work"]     = (
        (~hourly["dayofweek"].isin(WEEKEND)) &
        (hourly["hour_of_day"] >= WORK_START) &
        (hourly["hour_of_day"] <= WORK_END)
    ).astype(int)

    print(f"  Hourly rows   : {len(hourly):,}")
    print(f"  NaN (coverage): {hourly['energy_kwh'].isna().sum():,}")
    return hourly


def clean_for_training(hourly):
    """
    CLEAN PATH — full cleaning for autoencoder training.
    Removes outliers, imputes missing values, smooths.
    Result: the model learns ONLY from normal behaviour.
    """
    print("\n  [CLEAN path] IQR detection → imputation → smoothing...")

    # IQR contextual outlier detection (per season / hour / work context)
    for (season, is_work, is_we, hr), group in hourly.groupby(
            ["season","is_work","is_weekend","hour_of_day"]):
        vals = group["energy_kwh"].dropna()
        if len(vals) < 5: continue
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr     = q3 - q1
        bad_idx = group.index[
            group["energy_kwh"].notna() &
            ((group["energy_kwh"] > q3 + IQR_K * iqr) |
             (group["energy_kwh"] < max(0, q1 - IQR_K * iqr)))
        ]
        hourly.loc[bad_idx, "energy_kwh"] = np.nan

    # Zero-streak detection (> 4 consecutive zero hours flagged as NaN)
    is_zero  = hourly["energy_kwh"] == 0
    zero_grp = (is_zero != is_zero.shift()).cumsum()
    run_len  = is_zero.groupby(zero_grp).transform("sum")
    hourly.loc[is_zero & (run_len > ZERO_H), "energy_kwh"] = np.nan

    # Tier 1: linear interpolation for short gaps (≤ 3h)
    hourly["energy_kwh"] = hourly["energy_kwh"].interpolate(
        method="linear", limit=3)

    # Tier 2 & 3: context-aware imputation for longer gaps
    rng      = np.random.default_rng(42)
    nan_mask = hourly["energy_kwh"].isna()
    nan_grp  = (nan_mask != nan_mask.shift()).cumsum()
    gap_len  = nan_mask.groupby(nan_grp).transform("sum")

    for idx in hourly[hourly["energy_kwh"].isna()].index:
        row     = hourly.loc[idx]
        season, hr = row["season"], int(row["hour_of_day"])
        is_work = row["is_work"]; dt_val = row["datetime"]

        if gap_len.loc[idx] <= 24:        # Tier 2: ±14-day context window
            w_s = dt_val - pd.Timedelta(days=14)
            w_e = dt_val + pd.Timedelta(days=14)
            cands = hourly[
                (hourly["datetime"] >= w_s) & (hourly["datetime"] <= w_e) &
                (hourly["hour_of_day"] == hr) & (hourly["season"] == season) &
                (hourly["is_work"] == is_work) & (hourly["energy_kwh"].notna())
            ]["energy_kwh"]
        else:                              # Tier 3: global same-hour same-season
            cands = hourly[
                (hourly["hour_of_day"] == hr) & (hourly["season"] == season) &
                (hourly["energy_kwh"].notna())
            ]["energy_kwh"]

        if len(cands) < 3:
            cands = hourly[(hourly["hour_of_day"] == hr) &
                           hourly["energy_kwh"].notna()]["energy_kwh"]
        if len(cands) == 0: continue
        mu    = cands.mean()
        sigma = max(cands.std() if len(cands) > 1 else mu * 0.15, 0.01)
        hourly.at[idx, "energy_kwh"] = max(0.0, float(rng.normal(mu, sigma * 0.8)))

    # Gaussian smoothing (σ=0.8) then rolling Z-score cleanup (window=168h, z>3.5)
    hourly["energy_kwh"] = gaussian_filter1d(
        hourly["energy_kwh"].fillna(0).clip(lower=0).values, sigma=0.8
    ).clip(0)
    r_mean = hourly["energy_kwh"].rolling(168, center=True, min_periods=24).mean()
    r_std  = hourly["energy_kwh"].rolling(168, center=True, min_periods=24).std()
    r_med  = hourly["energy_kwh"].rolling(168, center=True, min_periods=24).median()
    z      = (hourly["energy_kwh"] - r_mean) / r_std.replace(0, np.nan)
    hourly.loc[z.abs() > 3.5, "energy_kwh"] = r_med[z.abs() > 3.5]
    hourly["energy_kwh"] = hourly["energy_kwh"].clip(lower=0)

    print(f"  NaN remaining  : {hourly['energy_kwh'].isna().sum()}")
    print(f"  Energy range   : {hourly['energy_kwh'].min():.4f} – "
          f"{hourly['energy_kwh'].max():.4f} kWh")
    return hourly


def prepare_for_detection(hourly):
    """
    RAW PATH — minimal processing for anomaly detection.
    Only fills true coverage gaps via forward-fill.
    Real anomalies (spikes, unusual consumption) are intentionally kept.
    """
    print("\n  [RAW path]   ffill coverage gaps only — anomalies preserved...")
    hourly["energy_kwh"] = hourly["energy_kwh"].ffill().bfill().clip(lower=0)
    print(f"  NaN remaining  : {hourly['energy_kwh'].isna().sum()}")
    print(f"  Energy range   : {hourly['energy_kwh'].min():.4f} – "
          f"{hourly['energy_kwh'].max():.4f} kWh")
    return hourly


def run():
    print("=" * 60)
    print("  STEP 1 — ETL  (two-path)")
    print("=" * 60)

    df_base  = load_and_aggregate(RAW_CSV)
    df_clean = clean_for_training(df_base.copy())
    df_raw   = prepare_for_detection(df_base.copy())

    cols = ["sn","datetime","energy_kwh","hour_of_day","dayofweek",
            "month","season","is_weekend","is_work"]
    df_clean[cols].to_csv(CLEAN_CSV, index=False)
    df_raw[cols].to_csv(RAW_OUT,     index=False)

    print(f"\n  ✅ Saved → {CLEAN_CSV}  shape={df_clean[cols].shape}")
    print(f"  ✅ Saved → {RAW_OUT}    shape={df_raw[cols].shape}")
    return df_clean[cols], df_raw[cols]

if __name__ == "__main__":
    run()
