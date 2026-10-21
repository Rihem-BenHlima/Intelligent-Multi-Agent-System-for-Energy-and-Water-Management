r"""
step1_etl_water.py — ETL Pipeline (WATER)
Input : dataset/water_consumption_office.csv (auto-detected, see find_water_csv())
Output: etl_output/etl_clean.csv  — cleaned, for training
        etl_output/etl_raw.csv    — minimal processing, anomalies preserved, for detection
Water is already hourly; no zero-streak rule (59% zeros is normal for offices);
no Gaussian smoothing (would erase true zeros); context = hour_of_day only.
Run: python step1_etl_water.py  [optional: path/to/file.csv]
"""

import sys, os, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

import config

# ── CONFIG ────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent

CLEAN_CSV  = config.CLEAN_CSV
RAW_OUT    = config.RAW_CSV
os.makedirs(config.ETL_OUTPUT_DIR, exist_ok=True)

SPIKE_Q    = 0.9995     # global hardware-spike cap (no season split)
IQR_K      = 3.5

# Known filename(s) to look for during auto-search, in priority order.
KNOWN_FILENAMES = ["water_consumption_office.csv", "Panther_office_Larry_water_m3.csv"]
GLOB_PATTERNS   = ["*water*.csv", "*Water*.csv"]


def find_water_csv():
    """Resolve water CSV path: CLI arg → config.WATER_CSV → auto-search (known filename then *water*.csv glob)."""
    # 1. CLI override
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        if p.exists():
            print(f"  Using path from command line: {p}")
            return str(p)
        print(f"  ⚠ Path given on command line does not exist: {p}")

    # 2. config.py override
    cfg_path = str(getattr(config, "WATER_CSV", "")).strip()
    if cfg_path:
        p = Path(cfg_path)
        if p.exists():
            print(f"  Using path from config.py: {p}")
            return str(p)
        print(f"  ⚠ config.py WATER_CSV path does not exist: {p}")

    # 3. Auto-search: dataset folder, script's own folder, current working directory
    dataset_dir = Path(config.DATASET_DIR)
    search_dirs = [dataset_dir, dataset_dir / "bdg2_office_water", SCRIPT_DIR, Path.cwd()]
    candidates, seen = [], set()

    def add(p: Path):
        rp = p.resolve()
        if rp not in seen and rp.exists():
            seen.add(rp); candidates.append(p)

    for d in search_dirs:
        for name in KNOWN_FILENAMES:
            add(d / name)
    for d in search_dirs:
        for pat in GLOB_PATTERNS:
            for p in d.glob(pat):
                add(p)

    if len(candidates) == 1:
        print(f"  Auto-detected water CSV: {candidates[0]}")
        return str(candidates[0])
    elif len(candidates) > 1:
        print(f"  ⚠ Multiple candidate CSVs found — using the first match:")
        for c in candidates:
            print(f"      - {c}")
        print(f"  If this is wrong, set WATER_CSV in config.py instead.")
        return str(candidates[0])

    # Nothing found anywhere
    raise FileNotFoundError(
        "Could not find the water CSV automatically. Fix with any ONE of:\n"
        "  (a) put the CSV next to this script (or in the folder you run it from), or\n"
        "  (b) run:  python step1_etl_water.py \"C:\\path\\to\\your_file.csv\", or\n"
        "  (c) edit config.py and set WATER_CSV to the full path."
    )


def load_consumption_liters(csv_path):
    """Read water CSV and return with a `consumption_l` (L/h) column; auto-converts consumption_m3→L (×1000)."""
    df = pd.read_csv(csv_path, parse_dates=["datetime"])
    if "consumption_l" in df.columns:
        df["consumption_l"] = df["consumption_l"].astype(float)
    elif "consumption_m3" in df.columns:
        df["consumption_l"] = df["consumption_m3"].astype(float) * 1000.0
    else:
        raise KeyError(
            f"{csv_path} has neither 'consumption_l' nor 'consumption_m3' "
            f"column — found: {list(df.columns)}"
        )
    return df


def load_hourly(path):
    """Load hourly water data and add hour_of_day context. Already hourly — no aggregation."""
    print(f"  Loading: {path}")
    df = load_consumption_liters(path).sort_values("datetime")
    df = df.drop_duplicates(subset="datetime").reset_index(drop=True)

    hourly = df[["datetime"]].copy()
    hourly["consumption_l"] = df["consumption_l"]   # litres/hour!
    hourly["sn"] = "Panther_office_Larry"

    full_idx = pd.date_range(hourly["datetime"].min(),
                             hourly["datetime"].max(), freq="h")
    hourly = (hourly.set_index("datetime").reindex(full_idx)
                    .rename_axis("datetime").reset_index())
    hourly["sn"] = hourly["sn"].ffill()

    # context: hour_of_day only
    hourly["hour_of_day"] = hourly["datetime"].dt.hour
    hourly["dayofweek"]   = hourly["datetime"].dt.dayofweek   # metadata only

    n_nan  = hourly["consumption_l"].isna().sum()
    n_zero = (hourly["consumption_l"] == 0).sum()
    print(f"  Hourly rows   : {len(hourly):,}")
    print(f"  Date range    : {hourly['datetime'].min()} → {hourly['datetime'].max()}")
    print(f"  Missing (NaN) : {n_nan:,} ({n_nan/len(hourly)*100:.2f}%)")
    print(f"  Zero hours    : {n_zero:,} ({n_zero/len(hourly)*100:.1f}%)  ← normal for offices")

    # global spike cap
    global_cap = hourly["consumption_l"].quantile(SPIKE_Q)
    spikes = hourly["consumption_l"] > global_cap
    hourly.loc[spikes, "consumption_l"] = np.nan
    print(f"  Spikes capped : {spikes.sum():,} (> global {SPIKE_Q} quantile = {global_cap:.1f} L/h)")
    return hourly


def clean_for_training(hourly):
    """Outlier removal (hour-of-day IQR on non-zero values) + zero-inflated imputation. No smoothing."""
    print("\n  [CLEAN path] hour-of-day IQR → imputation → rolling-Z cleanup...")

    # IQR on non-zero values only — 59% zeros make Q1=0 in most groups
    for hr, group in hourly.groupby("hour_of_day"):
        vals = group.loc[group["consumption_l"] > 0, "consumption_l"].dropna()
        if len(vals) < 5: continue
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr = q3 - q1
        hi  = q3 + IQR_K * iqr
        bad_idx = group.index[group["consumption_l"].notna() &
                              (group["consumption_l"] > hi)]
        hourly.loc[bad_idx, "consumption_l"] = np.nan

    print(f"  NaN after IQR  : {hourly['consumption_l'].isna().sum():,}")

    # Tier 1: linear interpolation for short gaps (≤ 3h)
    hourly["consumption_l"] = hourly["consumption_l"].interpolate(method="linear", limit=3)

    # context-aware imputation by hour_of_day
    rng      = np.random.default_rng(42)
    nan_mask = hourly["consumption_l"].isna()
    nan_grp  = (nan_mask != nan_mask.shift()).cumsum()
    gap_len  = nan_mask.groupby(nan_grp).transform("sum")

    for idx in hourly[hourly["consumption_l"].isna()].index:
        row = hourly.loc[idx]
        hr, dt_val = int(row["hour_of_day"]), row["datetime"]

        if gap_len.loc[idx] <= 24:  # short gap: ±14-day same-hour
            w_s, w_e = dt_val - pd.Timedelta(days=14), dt_val + pd.Timedelta(days=14)
            cands = hourly[
                (hourly["datetime"].between(w_s, w_e)) &
                (hourly["hour_of_day"] == hr) &
                hourly["consumption_l"].notna()]["consumption_l"]
        else:  # long gap: global same-hour
            cands = hourly[
                (hourly["hour_of_day"] == hr) &
                hourly["consumption_l"].notna()]["consumption_l"]

        if len(cands) < 3:
            cands = hourly[hourly["consumption_l"].notna()]["consumption_l"]
        if len(cands) == 0: continue

        # zero-inflated: sample zero with prob = context zero-share, else draw from non-zero dist
        p_zero = (cands == 0).mean()
        if rng.random() < p_zero:
            hourly.at[idx, "consumption_l"] = 0.0
        else:
            nz = cands[cands > 0]
            mu = nz.mean()
            sigma = max(nz.std() if len(nz) > 1 else mu * 0.15, 0.01)
            hourly.at[idx, "consumption_l"] = max(0.0, float(rng.normal(mu, sigma * 0.8)))

    # rolling Z cleanup on remaining extremes (168h window); replace outliers with rolling median
    e = hourly["consumption_l"].clip(lower=0)
    r_mean = e.rolling(168, center=True, min_periods=24).mean()
    r_std  = e.rolling(168, center=True, min_periods=24).std()
    r_med  = e.rolling(168, center=True, min_periods=24).median()
    z = (e - r_mean) / r_std.replace(0, np.nan)
    e[z.abs() > 3.5] = r_med[z.abs() > 3.5]
    hourly["consumption_l"] = e.clip(lower=0)

    print(f"  NaN remaining  : {hourly['consumption_l'].isna().sum()}")
    print(f"  Value range    : {hourly['consumption_l'].min():.2f} – "
          f"{hourly['consumption_l'].max():.2f} L/h")
    print(f"  Zero share     : {(hourly['consumption_l']==0).mean()*100:.1f}% (preserved)")
    return hourly


def prepare_for_detection(hourly, water_csv_path):
    """RAW path — restore spike-capped values from original, then ffill true gaps only. Anomalies preserved."""
    print("\n  [RAW path]   restore spikes + ffill missing only — anomalies preserved...")
    orig = load_consumption_liters(water_csv_path)
    orig = orig.drop_duplicates(subset="datetime").set_index("datetime")["consumption_l"]
    restored = hourly.set_index("datetime")["consumption_l"].copy()
    mask = restored.isna() & orig.reindex(restored.index).notna()
    restored[mask] = orig.reindex(restored.index)[mask]
    hourly["consumption_l"] = restored.values
    hourly["consumption_l"] = hourly["consumption_l"].ffill().bfill().clip(lower=0)
    print(f"  NaN remaining  : {hourly['consumption_l'].isna().sum()}")
    print(f"  Value range    : {hourly['consumption_l'].min():.2f} – "
          f"{hourly['consumption_l'].max():.2f} L/h")
    return hourly


def run():
    print("=" * 60)
    print("  STEP 1 — ETL  (WATER, two-path, no season/weekend context)")
    print("=" * 60)

    water_csv = find_water_csv()

    base     = load_hourly(water_csv)
    df_clean = clean_for_training(base.copy())
    df_raw   = prepare_for_detection(base.copy(), water_csv)

    cols = ["sn","datetime","consumption_l","hour_of_day","dayofweek"]
    df_clean[cols].to_csv(CLEAN_CSV, index=False)
    df_raw[cols].to_csv(RAW_OUT, index=False)
    print(f"\n  ✅ Saved → {CLEAN_CSV}  shape={df_clean[cols].shape}")
    print(f"  ✅ Saved → {RAW_OUT}    shape={df_raw[cols].shape}")
    print("  (column `consumption_l` holds LITRES/HOUR — use with")
    print("   step2_train_water.py / tune_thresholds_water.py / step3_detect_water.py)")
    print("  (dayofweek kept as metadata only — NOT used as a modeling context)")

if __name__ == "__main__":
    run()