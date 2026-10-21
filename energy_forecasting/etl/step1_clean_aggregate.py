"""
STEP 1 — Clean, Spike Removal & Hourly Aggregation
=====================================================
Cleans raw sub-minute energy readings, removes seasonal consumption
spikes, and aggregates readings into regular hourly intervals with
data coverage metrics.

Input : raw sub-minute readings (raw_data.csv)
Output: hourly energy with coverage flags (step1_hourly.csv)
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RAW_INPUT, STEP1_OUTPUT, STEP1_CFG, SEASONS, WEEKEND

SPIKE_Q = STEP1_CFG["spike_quantile"]
COV_THRESH = STEP1_CFG["coverage_threshold"]
MIN_READ = STEP1_CFG["min_readings"]


def get_season(month):
    for name, months in SEASONS.items():
        if month in months:
            return name
    return "autumn"


def run():
    print("=" * 60)
    print("  STEP 1 — Clean, Spike Removal & Hourly Aggregation")
    print("=" * 60)

    # 1. Load
    print(f"\n  Loading: {RAW_INPUT}")
    df = pd.read_csv(RAW_INPUT)
    df["date"] = pd.to_datetime(df["date"])
    print(f"  Raw rows: {len(df):,}")
    print(f"  Range: {df['date'].min()} → {df['date'].max()}")

    # 2. Sort & deduplicate
    before = len(df)
    df = df.drop_duplicates().sort_values("date").reset_index(drop=True)
    df = df.drop_duplicates(subset=["sn", "date"], keep="first")
    print(f"  Removed: {before - len(df):,} duplicates → {len(df):,} rows")

    # 3. Seasonal spike removal
    df["month"] = df["date"].dt.month
    df["season"] = df["month"].apply(get_season)

    global_cap = df["consumption_diff"].quantile(SPIKE_Q)
    seasonal_caps = {
        s: max(df[df["season"] == s]["consumption_diff"].quantile(SPIKE_Q), global_cap * 0.8)
        for s in df["season"].unique()
    }

    print(f"\n  Spike caps (kWh/reading):")
    for s, c in sorted(seasonal_caps.items()):
        print(f"    {s}: {c:.4f}")

    df["cap"] = df["season"].map(seasonal_caps)
    spikes = df["consumption_diff"] > df["cap"]
    print(f"  Spikes flagged: {spikes.sum():,}")
    df.loc[spikes, "consumption_diff"] = np.nan

    # 4. Hourly aggregation with dynamic interval
    df["dt"] = df["date"].diff().dt.total_seconds()
    df["hour"] = df["date"].dt.floor("h")

    hourly = (
        df.groupby("hour")
        .agg(
            sn=("sn", "first"),
            energy_kwh=("consumption_diff", "sum"),
            valid_readings=("consumption_diff", "count"),
            median_dt=("dt", "median"),
        )
        .reset_index()
        .rename(columns={"hour": "datetime"})
    )

    hourly["median_dt"] = hourly["median_dt"].fillna(22).replace(0, 22)
    hourly["expected"] = (3600 / hourly["median_dt"]).round(2)
    hourly["coverage"] = hourly["valid_readings"] / hourly["expected"]

    print(f"\n  Hourly rows: {len(hourly):,}")
    print(f"  Coverage: {hourly['coverage'].describe().round(3).to_string()}")

    # 5. Complete timeline & flag low coverage
    full_idx = pd.date_range(
        start=hourly["datetime"].min(),
        end=hourly["datetime"].max(),
        freq="h",
    )
    print(f"\n  Expected slots: {len(full_idx):,}")
    print(f"  Actual slots: {len(hourly):,}")
    print(f"  Gaps: {len(full_idx) - len(hourly):,}")

    hourly = hourly.set_index("datetime").reindex(full_idx)
    hourly.index.name = "datetime"
    hourly["sn"] = hourly["sn"].ffill()

    bad = (hourly["coverage"] < COV_THRESH) | (hourly["valid_readings"] < MIN_READ)
    bad = bad.fillna(True)
    hourly.loc[bad, "energy_kwh"] = np.nan

    print(f"  Bad hours blanked: {bad.sum():,}")
    print(f"  Total NaN: {hourly['energy_kwh'].isna().sum():,}")

    # 6. Add context columns
    hourly = hourly.reset_index()
    hourly["hour"] = hourly["datetime"].dt.hour
    hourly["dayofweek"] = hourly["datetime"].dt.dayofweek
    hourly["month"] = hourly["datetime"].dt.month
    hourly["is_weekend"] = hourly["dayofweek"].isin(WEEKEND).astype(int)

    # Drop helpers
    hourly = hourly.drop(columns=["median_dt", "expected", "valid_readings"])

    hourly.to_csv(STEP1_OUTPUT, index=False)
    print(f"\n  Saved → {STEP1_OUTPUT}")
    print(f"  Shape: {hourly.shape}")

    return hourly


if __name__ == "__main__":
    run()