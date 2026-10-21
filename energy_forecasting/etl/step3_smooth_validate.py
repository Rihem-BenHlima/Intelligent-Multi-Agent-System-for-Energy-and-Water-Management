"""
STEP 3 — Smooth & Final Validation
=====================================
Applies rolling z-score smoothing, repairs profile-level anomalies,
and runs final integrity checks (non-negative, no NaNs, full time coverage).

Input : imputed hourly energy (step2_imputed.csv)
Output: final clean dataset (step3_final.csv)
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import STEP2_OUTPUT, STEP3_OUTPUT, STEP3_CFG, SEASONS, WEEKEND, WORK_START, WORK_END

Z_THRESH = STEP3_CFG["z_threshold"]
WINDOW = STEP3_CFG["rolling_window"]
MIN_PER = STEP3_CFG["min_periods"]
PROF_Z = STEP3_CFG["profile_z"]
PROF_NOISE = STEP3_CFG["profile_noise"]
PROF_SIGMA = STEP3_CFG["profile_sigma"]


def get_season(month):
    for name, months in SEASONS.items():
        if month in months:
            return name
    return "autumn"


def run(step2_df=None):
    """Smooth and validate final energy series. Dual-use:
      - run()          → reads STEP2_OUTPUT csv, writes STEP3_OUTPUT csv (training)
      - run(step2_df)  → operates on given DataFrame, returns it without file I/O
    """
    print("=" * 60)
    print("  STEP 3 — Smooth & Final Validation")
    print("=" * 60)

    # 1. Load
    _from_disk = step2_df is None
    df = (pd.read_csv(STEP2_OUTPUT, parse_dates=["datetime"]) if _from_disk
          else step2_df.copy())
    df = df.sort_values("datetime").reset_index(drop=True)
    print(f"\n  Rows: {len(df):,}")

    # 2. Rolling z-score smoothing
    print("\n  --- Rolling z-score smoothing ---")
    r_mean = df["energy_kwh"].rolling(WINDOW, center=True, min_periods=MIN_PER).mean()
    r_std = df["energy_kwh"].rolling(WINDOW, center=True, min_periods=MIN_PER).std()
    r_med = df["energy_kwh"].rolling(WINDOW, center=True, min_periods=MIN_PER).median()

    z = (df["energy_kwh"] - r_mean) / r_std.replace(0, np.nan)
    mask = z.abs() > Z_THRESH
    print(f"  Z outliers (|z| > {Z_THRESH}): {mask.sum():,}")
    df.loc[mask, "energy_kwh"] = r_med[mask]

    # 3. Profile repair for remaining anomalies
    print("\n  --- Profile repair ---")
    df["date_d"] = df["datetime"].dt.date
    df["hour"] = df["datetime"].dt.hour

    prof = df.groupby(["season", "is_work_hours", "is_weekend", "hour"])["energy_kwh"].agg(["median", "std", "count"]).reset_index()
    prof.columns = ["season", "is_work_hours", "is_weekend", "hour", "p_med", "p_std", "p_n"]
    prof = prof[prof["p_n"] >= 3]
    prof["p_std"] = prof["p_std"].fillna(prof["p_med"] * 0.10).clip(lower=0.05)

    df = df.merge(prof, on=["season", "is_work_hours", "is_weekend", "hour"], how="left")

    hour_med = df.groupby("hour")["energy_kwh"].median().to_dict()
    df["p_med"] = df["p_med"].fillna(df["hour"].map(hour_med))
    df["p_std"] = df["p_std"].fillna(0.30)

    df["z_prof"] = (df["energy_kwh"] - df["p_med"]) / df["p_std"]
    anom = (df["z_prof"].abs() > PROF_Z) & (df["energy_kwh"].notna())
    print(f"  Profile anomalies: {anom.sum():,}")

    rng = np.random.default_rng(42)
    for idx in df[anom].index:
        mu = df.at[idx, "p_med"]
        sigma = max(df.at[idx, "p_std"] * PROF_NOISE, 0.02)
        df.at[idx, "energy_kwh"] = max(0.0, float(rng.normal(mu, sigma)))

    # Smooth repaired days
    rep_dates = set(df.loc[df[anom].index, "date_d"])
    for d in sorted(rep_dates):
        mask = df["date_d"] == d
        idx = df[mask].index
        if len(idx) < 3:
            continue
        vals = df.loc[idx, "energy_kwh"].values.astype(float)
        df.loc[idx, "energy_kwh"] = np.clip(gaussian_filter1d(vals, sigma=PROF_SIGMA), 0, None)

    # 4. Final validation
    print("\n  --- Final Validation ---")
    nan_c = df["energy_kwh"].isna().sum()
    neg_c = (df["energy_kwh"] < 0).sum()

    full_idx = pd.date_range(df["datetime"].min(), df["datetime"].max(), freq="h")
    missing = len(set(full_idx) - set(df["datetime"]))

    print(f"  NaN: {nan_c} {'✓' if nan_c == 0 else '✗'}")
    print(f"  Negative: {neg_c} {'✓' if neg_c == 0 else '✗'}")
    print(f"  Missing slots: {missing} {'✓' if missing == 0 else '✗'}")

    print(f"\n  Stats:")
    print(df["energy_kwh"].describe().round(3).to_string())

    print("\n  Mean by hour:")
    print(df.groupby("hour")["energy_kwh"].mean().round(3).to_string())

    print("\n  Mean by season:")
    for s in ["summer", "winter", "autumn", "spring"]:
        m = df[df["season"] == s]["energy_kwh"].mean()
        print(f"    {s}: {m:.3f} kWh")

    print("\n  Work hours vs off hours:")
    print(df.groupby("is_work_hours")["energy_kwh"].mean().round(3).to_string())

    # 5. Export
    out = df[["sn", "datetime", "energy_kwh", "hour", "dayofweek", "month",
              "is_weekend", "is_work_hours", "season"]].copy()
    if _from_disk:
        out.to_csv(STEP3_OUTPUT, index=False)
        print(f"\n  Saved → {STEP3_OUTPUT}")
    print(f"  Shape: {out.shape}")

    return out


if __name__ == "__main__":
    run()