"""
STEP 2 — Detect Outliers & Impute Missing Values
===================================================
Identifies contextual outliers and missing values across seasonal
and work-hour regimes, then applies multi-tier imputation:
  - Tier 1: Linear interpolation for short gaps (<= 3h)
  - Tier 2: Local contextual window matching for medium gaps (<= 24h)
  - Tier 3: Global seasonal profile matching for long gaps (> 24h)

Input : hourly energy with gaps (step1_hourly.csv)
Output: fully imputed hourly energy (step2_imputed.csv)
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import hashlib
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import STEP1_OUTPUT, STEP2_OUTPUT, STEP2_CFG
from config import WORK_START, WORK_END, WEEKEND, SEASONS, HOLIDAY_SET

IQR_K = STEP2_CFG["outlier_iqr_k"]
ZERO_H = STEP2_CFG["zero_streak_hours"]
TIER1 = STEP2_CFG["tier1_limit"]
T2_WIN = STEP2_CFG["tier2_window_days"]
T2_MIN = STEP2_CFG["tier2_min_cands"]
T3_MIN = STEP2_CFG["tier3_min_cands"]
NOISE = STEP2_CFG["noise_factor"]
SIGMA = STEP2_CFG["smooth_sigma"]


def get_season(month):
    for name, months in SEASONS.items():
        if month in months:
            return name
    return "autumn"


def is_work_hours(row):
    """Check if a timestamp falls during scheduled work hours:
    Mon–Fri, WORK_START..WORK_END inclusive, excluding public holidays."""
    return (
        row["dayofweek"] not in WEEKEND
        and WORK_START <= row["hour"] <= WORK_END
        and row["date_norm"] not in HOLIDAY_SET
    )


def stable_seed(seed_val):
    """Generate a deterministic seed across Python processes."""
    digest = hashlib.sha256(str(seed_val).encode("utf-8")).hexdigest()
    return int(digest, 16) % (2**31)


def impute_noise(candidates, seed_val):
    if len(candidates) == 0:
        return np.nan
    mu = candidates.mean()
    sigma = candidates.std() if len(candidates) > 1 else mu * 0.15
    sigma = max(sigma, 0.01)
    rng = np.random.default_rng(stable_seed(seed_val))
    return max(0.0, mu + rng.normal(0, sigma * NOISE))


def run(step1_df=None):
    """Detect outliers and impute missing values. Dual-use:
      - run()            → reads STEP1_OUTPUT csv, writes STEP2_OUTPUT csv (training)
      - run(step1_df)    → operates on given DataFrame, returns it without file I/O
    """
    print("=" * 60)
    print("  STEP 2 — Detect Outliers & Impute")
    print("=" * 60)

    # 1. Load
    _from_disk = step1_df is None
    df = (pd.read_csv(STEP1_OUTPUT, parse_dates=["datetime"]) if _from_disk
          else step1_df.copy())
    df = df.sort_values("datetime").reset_index(drop=True)
    print(f"\n  Rows: {len(df):,}")
    print(f"  Range: {df['datetime'].min()} → {df['datetime'].max()}")
    print(f"  Initial NaN: {df['energy_kwh'].isna().sum():,}")

    # 2. Build context columns
    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["season"] = df["month"].apply(get_season)
    df["is_weekend"] = df["dayofweek"].isin(WEEKEND).astype(int)
    # Treat public holidays falling on weekdays as non-work periods
    df["date_norm"] = df["datetime"].dt.normalize()
    df["is_work_hours"] = df.apply(is_work_hours, axis=1).astype(int)
    df = df.drop(columns=["date_norm"])

    # 3. DETECT: Contextual outliers
    print("\n  --- Detection ---")
    outlier_mask = pd.Series(False, index=df.index)

    for (season, is_work, is_we, hr), group in df.groupby(["season", "is_work_hours", "is_weekend", "hour"]):
        vals = group["energy_kwh"].dropna()
        if len(vals) < 5:
            continue
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr = q3 - q1
        upper = q3 + IQR_K * iqr
        lower = max(0, q1 - IQR_K * iqr)
        bad = group.index[group["energy_kwh"].notna() & ((group["energy_kwh"] > upper) | (group["energy_kwh"] < lower))]
        outlier_mask.loc[bad] = True

    print(f"  Outliers flagged: {outlier_mask.sum():,}")
    df.loc[outlier_mask, "energy_kwh"] = np.nan

    # 4. DETECT: Long zero streaks
    is_zero = df["energy_kwh"] == 0
    zero_grp = (is_zero != is_zero.shift()).cumsum()
    run_len = is_zero.groupby(zero_grp).transform("sum")
    long_zero = is_zero & (run_len > ZERO_H)

    # Suspicious zeros during work hours
    susp_zero = (df["energy_kwh"] == 0) & (df["is_work_hours"] == 1) & (~long_zero)

    print(f"  Long zero streaks: {long_zero.sum():,}")
    print(f"  Suspicious work-hour zeros: {susp_zero.sum():,}")
    df.loc[long_zero | susp_zero, "energy_kwh"] = np.nan

    total_nan = df["energy_kwh"].isna().sum()
    print(f"  Total NaN to impute: {total_nan:,}")

    # 5. IMPUTE: Classify gaps
    nan_mask = df["energy_kwh"].isna()
    nan_grp = (nan_mask != nan_mask.shift()).cumsum()
    gap_len = nan_mask.groupby(nan_grp).transform("sum")

    short = nan_mask & (gap_len <= TIER1)
    medium = nan_mask & (gap_len > TIER1) & (gap_len <= 24)
    long = nan_mask & (gap_len > 24)

    print(f"\n  Gaps: short={short.sum():,}, medium={medium.sum():,}, long={long.sum():,}")

    # 6. IMPUTE: Tier 1 — linear interpolation
    df["energy_kwh"] = df["energy_kwh"].interpolate(method="linear", limit=TIER1)
    print(f"  After Tier 1: {df['energy_kwh'].isna().sum():,} NaN")

    # 7. IMPUTE: Tier 2 — medium gaps with context
    filled_t2 = 0
    for idx in df[medium & df["energy_kwh"].isna()].index:
        row = df.loc[idx]
        season, hr = row["season"], int(row["hour"])
        is_work, is_we = row["is_work_hours"], row["is_weekend"]
        dt_val = row["datetime"]
        w_start = dt_val - pd.Timedelta(days=T2_WIN)
        w_end = dt_val + pd.Timedelta(days=T2_WIN)

        cands = df[
            (df["datetime"] >= w_start) & (df["datetime"] <= w_end) &
            (df["hour"] == hr) & (df["season"] == season) &
            (df["is_work_hours"] == is_work) & (df["energy_kwh"].notna())
        ]["energy_kwh"]

        if len(cands) < T2_MIN:
            cands = df[
                (df["datetime"] >= w_start) & (df["datetime"] <= w_end) &
                (df["hour"] == hr) & (df["season"] == season) & (df["energy_kwh"].notna())
            ]["energy_kwh"]

        val = impute_noise(cands, (str(dt_val.date()), hr, "t2"))
        if not np.isnan(val):
            df.at[idx, "energy_kwh"] = val
            filled_t2 += 1

    print(f"  Tier 2 filled: {filled_t2:,}")

    # 8. IMPUTE: Tier 3 — long gaps with global context
    filled_t3 = 0
    imp_dates = set()

    for idx in df[long & df["energy_kwh"].isna()].index:
        row = df.loc[idx]
        season, hr = row["season"], int(row["hour"])
        is_work = row["is_work_hours"]
        date_str = str(row["datetime"].date())
        imp_dates.add(row["datetime"].date())

        cands = df[
            (df["hour"] == hr) & (df["season"] == season) &
            (df["is_work_hours"] == is_work) & (df["energy_kwh"].notna())
        ]["energy_kwh"]

        if len(cands) < T3_MIN:
            cands = df[
                (df["hour"] == hr) & (df["season"] == season) & (df["energy_kwh"].notna())
            ]["energy_kwh"]

        if len(cands) < T3_MIN:
            cands = df[(df["hour"] == hr) & (df["energy_kwh"].notna())]["energy_kwh"]

        val = impute_noise(cands, (date_str, hr, "t3"))
        if not np.isnan(val):
            df.at[idx, "energy_kwh"] = val
            filled_t3 += 1

    print(f"  Tier 3 filled: {filled_t3:,}")

    # 9. Smooth imputed days
    df["date_d"] = df["datetime"].dt.date
    for d in sorted(imp_dates):
        mask = df["date_d"] == d
        idx = df[mask].index
        if len(idx) < 3:
            continue
        vals = df.loc[idx, "energy_kwh"].values.copy()
        df.loc[idx, "energy_kwh"] = np.clip(gaussian_filter1d(vals, sigma=SIGMA), 0, None)

    df["energy_kwh"] = df["energy_kwh"].clip(lower=0)

    # 10. Fill any remaining
    remaining = df["energy_kwh"].isna().sum()
    if remaining > 0:
        df["energy_kwh"] = df["energy_kwh"].ffill().bfill()

    print(f"\n  Final NaN: {df['energy_kwh'].isna().sum()}")
    print(f"  Range: {df['energy_kwh'].min():.3f} - {df['energy_kwh'].max():.3f} kWh")

    # Validation
    print("\n  Seasonal means (work hours):")
    for s in ["summer", "winter", "autumn", "spring"]:
        m = df[(df["season"] == s) & (df["is_work_hours"] == 1)]["energy_kwh"].mean()
        print(f"    {s}: {m:.3f} kWh")

    # Save
    out = df[["sn", "datetime", "energy_kwh", "hour", "dayofweek", "month",
              "is_weekend", "is_work_hours", "season"]].copy()
    if _from_disk:
        out.to_csv(STEP2_OUTPUT, index=False)
        print(f"\n  Saved → {STEP2_OUTPUT}")
    print(f"  Shape: {out.shape}")

    return out


if __name__ == "__main__":
    run()