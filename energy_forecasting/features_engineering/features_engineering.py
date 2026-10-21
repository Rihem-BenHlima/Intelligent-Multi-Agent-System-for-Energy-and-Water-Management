"""
features_engineering.py
========================
Feature engineering for 24h-ahead building energy forecasting.

Features generated:
  - Autoregressive lags (1h to 336h)
  - Rolling and exponentially weighted statistics (mean, std, min, max, velocity)
  - Cyclical and binary calendar indicators for target hour (t+24)
  - Observed weather variables and thermal demand proxies (CDD/HDD, heat index)
  - 24h-ahead weather forecast inputs from Open-Meteo
  - Cross-feature interactions and expanding-window historical consumption profiles
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
from pathlib import Path

from config import (
    INPUT_CSV,
    WEATHER_CSV,
    FORECAST_CSV,
    FEATURES_CSV  as OUTPUT_CSV,
    LATITUDE, LONGITUDE, TIMEZONE, HORIZON,
    TUNISIA_HOLIDAYS, HOLIDAY_SET,
    CDD_THRESHOLDS, HDD_THRESHOLDS,
    TRAIN_RATIO,
    validate_paths,
)

# Summer months (Tunisia: peak AC load, hardest to forecast)
SUMMER_MONTHS = {6, 7, 8, 9}


# ─────────────────────────────────────────────────────────────
# 1. WEATHER DOWNLOAD
# ─────────────────────────────────────────────────────────────
def _missing_ranges(existing: pd.DataFrame, start, end):
    """Gaps between what `existing` already covers and [start, end]."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if existing is None or existing.empty:
        return [(start, end)]
    lo, hi = existing["datetime"].min(), existing["datetime"].max()
    ranges = []
    if start < lo:
        ranges.append((start, min(lo - pd.Timedelta(hours=1), end)))
    if end > hi:
        ranges.append((max(hi + pd.Timedelta(hours=1), start), end))
    return ranges


def _fetch_weather_chunks(start, end) -> pd.DataFrame:
    """Download hourly weather from Open-Meteo archive API in 6-month chunks."""
    chunks = []
    current = pd.Timestamp(start)
    end     = pd.Timestamp(end)

    # Clip requested end_date to current date to avoid archive API 400 errors
    api_limit = pd.Timestamp.now().normalize()
    if end > api_limit:
        print(f"    (clipping requested end {end.date()} → {api_limit.date()}: "
              f"archive API has no data beyond real 'today')")
        end = api_limit
    if current > end:
        return pd.DataFrame()

    while current <= end:
        chunk_end = min(current + pd.DateOffset(months=6), end)
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude":   LATITUDE,
            "longitude":  LONGITUDE,
            "start_date": current.strftime("%Y-%m-%d"),
            "end_date":   chunk_end.strftime("%Y-%m-%d"),
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
                "apparent_temperature",
                "dew_point_2m",
                "cloud_cover",
                "wind_speed_10m",
                "precipitation",
                "shortwave_radiation",
            ],
            "timezone": TIMEZONE,
        }
        try:
            resp = requests.get(url, params=params, timeout=90)
            resp.raise_for_status()
            data = resp.json()["hourly"]
            chunk_df = pd.DataFrame({
                "datetime":             pd.to_datetime(data["time"]),
                "temperature":          data["temperature_2m"],
                "humidity":             data["relative_humidity_2m"],
                "apparent_temperature": data["apparent_temperature"],
                "dew_point":            data["dew_point_2m"],
                "cloud_cover":          data["cloud_cover"],
                "wind_speed":           data["wind_speed_10m"],
                "precipitation":        data["precipitation"],
                "solar_radiation":      data["shortwave_radiation"],
            })
            if chunk_df["datetime"].dt.tz is not None:
                chunk_df["datetime"] = chunk_df["datetime"].dt.tz_localize(None)
            chunks.append(chunk_df)
            print(f"    ✓ {current.strftime('%Y-%m-%d')} → {chunk_end.strftime('%Y-%m-%d')} "
                  f"({len(chunk_df):,} rows)")
        except Exception as ex:
            print(f"    ✗ Failed chunk {current.strftime('%Y-%m-%d')}: {ex}")

        current = chunk_end + pd.Timedelta(days=1)

    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def download_weather(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Download hourly weather from Open-Meteo archive API, reusing WEATHER_CSV
    as a cache and only requesting missing date ranges.
    """
    cached = pd.DataFrame()
    if WEATHER_CSV.exists():
        cached = pd.read_csv(WEATHER_CSV, parse_dates=["datetime"])
        if cached["datetime"].dt.tz is not None:
            cached["datetime"] = cached["datetime"].dt.tz_localize(None)
        print(f"  [Weather] Cached: {WEATHER_CSV.name} ({len(cached):,} rows, "
              f"{cached['datetime'].min().date()} → {cached['datetime'].max().date()})")

    missing = _missing_ranges(cached, start_date, end_date)
    if not missing:
        print(f"  [Weather] Cache already covers {start_date} → {end_date}")
        return cached

    for s, e in missing:
        print(f"  [Weather] Cache gap detected: downloading {s.date()} → {e.date()}...")
        new = _fetch_weather_chunks(s, e)
        if not new.empty:
            cached = pd.concat([cached, new], ignore_index=True)

    if cached.empty:
        raise RuntimeError("Weather download failed for all chunks.")

    wdf = (cached.drop_duplicates("datetime")
                 .sort_values("datetime")
                 .reset_index(drop=True))

    wdf.to_csv(WEATHER_CSV, index=False)
    print(f"  [Weather] Saved → {WEATHER_CSV.name}  ({len(wdf):,} rows, "
          f"{wdf['datetime'].min().date()} → {wdf['datetime'].max().date()})")
    return wdf


# ─────────────────────────────────────────────────────────────
# 2. WEATHER FORECAST DOWNLOAD
# ─────────────────────────────────────────────────────────────
def download_weather_forecast_24h(start_date: str) -> pd.DataFrame:
    """
    Download historical 24h-ahead forecasts from Open-Meteo Previous Runs API,
    using FORECAST_CSV as a cache and updating when new dates are required.
    """
    cached = pd.DataFrame()
    if FORECAST_CSV.exists():
        cached = pd.read_csv(FORECAST_CSV, parse_dates=["datetime"])
        if cached["datetime"].dt.tz is not None:
            cached["datetime"] = cached["datetime"].dt.tz_localize(None)
        print(f"  [Weather forecast] Cached: {FORECAST_CSV.name} ({len(cached):,} rows, "
              f"{cached['datetime'].min().date()} → {cached['datetime'].max().date()})")

    today = pd.Timestamp.now().normalize()
    needed_start = pd.Timestamp(start_date)
    covers_start = not cached.empty and cached["datetime"].min() <= needed_start
    covers_end   = not cached.empty and cached["datetime"].max() >= today - pd.Timedelta(days=1)
    if covers_start and covers_end:
        print(f"  [Weather forecast] Cache already covers required range")
        return cached

    past_days = (today - needed_start).days + 3
    past_days = int(min(max(past_days, 1), 730))

    print(f"  [Weather forecast] Cache gap detected: re-downloading 24h-ahead forecasts "
          f"from Open-Meteo (past_days={past_days})...")

    variables = ["temperature_2m", "relative_humidity_2m", "cloud_cover", "shortwave_radiation"]
    url = "https://previous-runs-api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": variables + [f"{v}_previous_day1" for v in variables],
        "past_days": past_days,
        "forecast_days": 1,
        "timezone": TIMEZONE,
    }

    try:
        resp = requests.get(url, params=params, timeout=90)
        resp.raise_for_status()
        data = resp.json()["hourly"]
    except Exception as ex:
        raise RuntimeError(f"Weather forecast download failed: {ex}")

    fdf = pd.DataFrame({
        "datetime":      pd.to_datetime(data["time"]),
        "temp_24h":      data["temperature_2m_previous_day1"],
        "humidity_24h":  data["relative_humidity_2m_previous_day1"],
        "cloud_24h":     data["cloud_cover_previous_day1"],
        "solar_rad_24h": data["shortwave_radiation_previous_day1"],
    })
    if fdf["datetime"].dt.tz is not None:
        fdf["datetime"] = fdf["datetime"].dt.tz_localize(None)
    fdf = fdf.dropna(subset=["temp_24h"])

    if not cached.empty:
        fdf = pd.concat([cached, fdf], ignore_index=True)

    fdf = (fdf.drop_duplicates("datetime", keep="last")
              .sort_values("datetime")
              .reset_index(drop=True))

    fdf.to_csv(FORECAST_CSV, index=False)
    print(f"  [Weather forecast] Saved → {FORECAST_CSV.name}  ({len(fdf):,} rows, "
          f"{fdf['datetime'].min().date()} → {fdf['datetime'].max().date()})")
    return fdf


# ─────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING — LEAKAGE-FREE
# ─────────────────────────────────────────────────────────────
def build_features(df: pd.DataFrame, forecast_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Build all features from the merged energy+weather dataframe.

    Key rule: every feature at row t uses only information
    available at prediction time t (i.e., past energy & weather,
    future calendar, and weather forecast for t+24).
    """
    e = df["energy_kwh"]
    F = {}   # feature dict — assembled into DataFrame at the end

    # ── A. AUTOREGRESSIVE LAGS ────────────────────────────────
    print("  [1/9] Autoregressive lags...")
    for lag in [1, 2, 3, 6, 12, 24, 25, 48, 49, 72, 73, 96, 120, 144, 168, 336]:
        F[f"lag_{lag}h"] = e.shift(lag)

    # ── B. ROLLING STATISTICS ─────────────────────────────────
    print("  [2/9] Rolling statistics...")
    s1 = e.shift(1)
    for w in [3, 6, 12, 24, 48, 72, 168]:
        F[f"roll_mean_{w}h"] = s1.rolling(w, min_periods=1).mean()
        F[f"roll_std_{w}h"]  = s1.rolling(w, min_periods=1).std().fillna(0)

    F["roll_max_24h"]   = s1.rolling(24, min_periods=1).max()
    F["roll_min_24h"]   = s1.rolling(24, min_periods=1).min()
    F["roll_range_24h"] = F["roll_max_24h"] - F["roll_min_24h"]
    F["roll_max_48h"]   = s1.rolling(48, min_periods=1).max()   # summer: 2-day peak

    # EWM (exponentially weighted)
    F["ewm_24h"]  = s1.ewm(span=24,  adjust=False).mean()
    F["ewm_72h"]  = s1.ewm(span=72,  adjust=False).mean()
    F["ewm_168h"] = s1.ewm(span=168, adjust=False).mean()

    # Load velocity
    F["load_velocity"] = s1 - e.shift(2)

    # ── C. CALENDAR OF TARGET HOUR (t+24) ─────────────────────
    print("  [3/9] Calendar features (target hour = t+24)...")
    target_dt = df["datetime"] + pd.Timedelta(hours=HORIZON)

    th  = target_dt.dt.hour
    td  = target_dt.dt.dayofweek
    tm  = target_dt.dt.month
    doy = target_dt.dt.dayofyear

    F["hour_sin"]  = np.sin(2 * np.pi * th  / 24)
    F["hour_cos"]  = np.cos(2 * np.pi * th  / 24)
    F["hour_sin2"] = np.sin(4 * np.pi * th  / 24)
    F["hour_cos2"] = np.cos(4 * np.pi * th  / 24)
    F["dow_sin"]   = np.sin(2 * np.pi * td  / 7)
    F["dow_cos"]   = np.cos(2 * np.pi * td  / 7)
    F["month_sin"] = np.sin(2 * np.pi * tm  / 12)
    F["month_cos"] = np.cos(2 * np.pi * tm  / 12)
    F["doy_sin"]   = np.sin(2 * np.pi * doy / 365)
    F["doy_cos"]   = np.cos(2 * np.pi * doy / 365)

    F["t_hour"]  = th
    F["t_dow"]   = td
    F["t_month"] = tm
    F["t_doy"]   = doy

    # ── D. BINARY CALENDAR FLAGS (target hour) ────────────────
    print("  [4/9] Calendar flags...")
    F["is_weekend"] = (td >= 5).astype(int)
    F["is_holiday"] = target_dt.dt.normalize().isin(HOLIDAY_SET).astype(int)
    F["is_work"]    = (
        (th >= 8) & (th <= 18) &
        (td < 5) &
        (~target_dt.dt.normalize().isin(HOLIDAY_SET))
    ).astype(int)

    F["bridge_before"] = (target_dt + pd.Timedelta(days=1)).dt.normalize().isin(HOLIDAY_SET).astype(int)
    F["bridge_after"]  = (target_dt - pd.Timedelta(days=1)).dt.normalize().isin(HOLIDAY_SET).astype(int)

    F["is_morning_peak"] = ((th >= 8)  & (th <= 10)).astype(int)
    F["is_midday"]       = ((th >= 11) & (th <= 14)).astype(int)
    F["is_afternoon"]    = ((th >= 15) & (th <= 18)).astype(int)
    F["is_night"]        = ((th >= 22) | (th <= 5)).astype(int)

    # Summer regime flag — used in interactions below
    F["is_summer"] = tm.isin(SUMMER_MONTHS).astype(int)

    # ── E. WEATHER — OBSERVED AT T (shift(1)) ─────────────────
    print("  [5/9] Observed weather features...")

    if "temperature" in df.columns:
        t1  = df["temperature"].shift(1)
        h1  = df["humidity"].shift(1)
        cc1 = df["cloud_cover"].shift(1)
        dp1 = df["dew_point"].shift(1) if "dew_point" in df.columns else None

        F["temp_obs"]     = t1
        F["humidity_obs"] = h1
        F["cloud_obs"]    = cc1

        for thr in CDD_THRESHOLDS:
            F[f"cdd_{thr}"] = (t1 - thr).clip(lower=0)
        for thr in HDD_THRESHOLDS:
            F[f"hdd_{thr}"] = (thr - t1).clip(lower=0)

        F["cdd_28_sq"] = F["cdd_28"] ** 2

        F["temp_ema_24h"]  = t1.ewm(span=24,  adjust=False).mean()
        F["temp_ema_72h"]  = t1.ewm(span=72,  adjust=False).mean()
        F["temp_max_24h"]  = t1.rolling(24, min_periods=1).max()
        F["temp_min_24h"]  = t1.rolling(24, min_periods=1).min()
        F["temp_diurnal"]  = F["temp_max_24h"] - F["temp_min_24h"]
        F["temp_acc"]      = F["temp_ema_72h"] - F["temp_ema_24h"]

        F["is_hot_hour"]    = (t1 > 25).astype(int)
        F["consec_hot_72h"] = F["is_hot_hour"].rolling(72, min_periods=1).sum()

        F["cdd_x_hum"]            = F["cdd_28"] * (h1 / 100)
        F["humidity_rolling_24h"] = h1.rolling(24, min_periods=1).mean()

        if dp1 is not None:
            F["humidex"] = t1 + (5 / 9) * (
                6.1078 * np.exp(17.269 * dp1 / (237.3 + dp1)) - 10
            )
            F["dew_point_obs"] = dp1

        F["solar_proxy_obs"] = (1 - cc1 / 100) * t1

        if "solar_radiation" in df.columns:
            rad1 = df["solar_radiation"].shift(1)
            F["solar_rad_obs"]      = rad1
            F["solar_rad_24h_mean"] = rad1.rolling(24, min_periods=1).mean()

        # ── E-SUMMER. SUMMER-SPECIFIC OBSERVED FEATURES ───────
        # Extreme heat non-linearities, multi-day accumulation, and diurnal reset

        # 1. Non-linear extreme-heat features
        F["temp_sq"]     = t1 ** 2                          # quadratic heat surge
        F["above_35"]    = (t1 - 35).clip(lower=0)          # high-temperature threshold
        F["above_38"]    = (t1 - 38).clip(lower=0)          # extreme heat regime

        # 2. Multi-day heat accumulation
        F["temp_acc_72h"]     = t1.rolling(72,  min_periods=1).mean()  # 3-day mean
        F["temp_acc_168h"]    = t1.rolling(168, min_periods=1).mean()  # 7-day mean
        F["temp_acc_delta"]   = F["temp_acc_72h"] - F["temp_acc_168h"] # thermal accumulation trend

        # 3. Overnight thermal recovery (shifted by 1 day to ensure causal ordering)
        temp_df = pd.DataFrame({"temperature": t1.values}, index=df["datetime"])

        daily_max = temp_df.groupby(temp_df.index.date)["temperature"].max().sort_index()
        daily_min = temp_df.groupby(temp_df.index.date)["temperature"].min().sort_index()

        prev_day_max = daily_max.shift(1)  # previous day's max
        prev_day_min = daily_min.shift(1)  # previous day's min

        F["daily_max_temp"] = df["datetime"].dt.date.map(prev_day_max.to_dict())
        F["daily_min_temp"] = df["datetime"].dt.date.map(prev_day_min.to_dict())
        F["day_night_delta"] = F["daily_max_temp"] - F["daily_min_temp"]

        # Trailing 24h min ending at t-1
        F["night_min_24h"] = t1.rolling(24, min_periods=1).min()

        # 4. Apparent temperature / heat-index (feels-like)
        if dp1 is not None:
            # Steadman heat index approximation
            F["heat_index"] = (
                -8.78469475556
                + 1.61139411   * t1
                + 2.33854883889 * (h1 / 100)
                - 0.14611605   * t1 * (h1 / 100)
                - 0.012308094  * t1 ** 2
                - 0.016424828  * (h1 / 100) ** 2
                + 0.002211732  * t1 ** 2 * (h1 / 100)
                + 0.00072546   * t1 * (h1 / 100) ** 2
                - 0.000003582  * t1 ** 2 * (h1 / 100) ** 2
            )
        else:
            # Simplified: temp + humidity penalty
            F["heat_index"] = t1 + 0.33 * (h1 / 100) * (t1 - 14.4) - 4.0

        F["heat_index_sq"]    = F["heat_index"].clip(lower=0) ** 2
        F["hi_above_35"]      = (F["heat_index"] - 35).clip(lower=0)

        # 5. Consecutive hot days (>35°C over past 5 days, shifted by 1 day)
        is_hot_day = (daily_max > 35).astype(int)
        consec_hot_5d = is_hot_day.rolling(5, min_periods=1).sum().shift(1)
        F["consec_hot_days"] = df["datetime"].dt.date.map(consec_hot_5d.to_dict())


    # ── F. WEATHER — FORECAST AT T+24 ─────────────────────────
    # Uses 24h-ahead forecasts from Open-Meteo Previous Runs API aligned with target_dt
    print("  [6/9] Forecast weather at t+24...")

    if "temperature" in df.columns and forecast_df is not None and not forecast_df.empty:
        aligned = forecast_df.set_index("datetime").reindex(target_dt)
        aligned.index = df.index

        t24  = aligned["temp_24h"]
        h24  = aligned["humidity_24h"]
        cc24 = aligned["cloud_24h"]

        F["temp_24h"]     = t24
        F["humidity_24h"] = h24
        F["cloud_24h"]    = cc24

        for thr in CDD_THRESHOLDS:
            F[f"cdd_{thr}_24h"] = (t24 - thr).clip(lower=0)
        F["cdd_28_sq_24h"]  = F["cdd_28_24h"] ** 2
        for thr in HDD_THRESHOLDS:
            F[f"hdd_{thr}_24h"] = (thr - t24).clip(lower=0)

        # Summer-specific forecast features
        F["temp_sq_24h"]   = t24 ** 2
        F["above_35_24h"]  = (t24 - 35).clip(lower=0)
        F["above_38_24h"]  = (t24 - 38).clip(lower=0)

        F["solar_proxy_24h"] = (1 - cc24 / 100) * t24
        F["cdd_x_hum_24h"]   = F["cdd_28_24h"] * (h24 / 100)

        if "solar_rad_24h" in aligned.columns:
            F["solar_rad_24h"] = aligned["solar_rad_24h"]

    # ── G. INTERACTION FEATURES ────────────────────────────────
    print("  [7/9] Interaction features...")

    F["work_x_lag24"]  = F["is_work"] * F["lag_24h"]
    F["work_x_lag168"] = F["is_work"] * F["lag_168h"]
    F["we_x_lag24"]    = F["is_weekend"] * F["lag_24h"]
    F["we_x_lag168"]   = F["is_weekend"] * F["lag_168h"]

    if "temp_24h" in F:
        F["lag168_x_t24"]   = F["lag_168h"] * F["temp_24h"]
        F["lag24_x_t24"]    = F["lag_24h"]  * F["temp_24h"]
        F["work_x_cdd24"]   = F["is_work"]  * F["cdd_28_24h"]
        F["work_x_cdd_obs"] = F["is_work"]  * F["cdd_28"]

        # Summer-regime interactions — model should use different
        # lag patterns when it's hot vs not
        F["summer_x_lag24"]    = F["is_summer"] * F["lag_24h"]
        F["summer_x_lag168"]   = F["is_summer"] * F["lag_168h"]
        F["summer_x_cdd28_24"] = F["is_summer"] * F["cdd_28_24h"]
        F["summer_x_above35"]  = F["is_summer"] * F.get("above_35_24h",
                                    (F["temp_24h"] - 35).clip(lower=0))
        F["summer_x_hi"]       = F["is_summer"] * F.get("heat_index",
                                    F["temp_obs"])

    F["hour_x_month"] = F["t_hour"] * F["t_month"]
    F["work_x_month"] = F["is_work"] * F["t_month"]
    F["dow_x_hour"]   = F["t_dow"]  * F["t_hour"]

    F["momentum_24h"]  = F["lag_24h"]  - F["lag_48h"]
    F["momentum_168h"] = F["lag_168h"] - e.shift(336)

    # ── H. EXPANDING-WINDOW PROFILE ────────────────────────────
    print("  [8/9] Expanding-window profile feature (slow, ~60s)...")

    tgt_dow_arr  = td.values
    tgt_hour_arr = th.values
    is_work_arr  = F["is_work"].values
    energy_arr   = e.values
    prof         = np.full(len(df), np.nan)

    for i in range(336, len(df)):
        mask = (
            (tgt_dow_arr[:i]  == tgt_dow_arr[i])  &
            (tgt_hour_arr[:i] == tgt_hour_arr[i]) &
            (is_work_arr[:i]  == is_work_arr[i])
        )
        vals = energy_arr[:i][mask]
        if len(vals) >= 5:
            prof[i] = np.median(vals[-20:])

    F["profile_median"] = prof

    # ── I. SUMMER PROFILE (expanding, summer-only) ─────────────
    # A separate profile computed only over summer months —
    # avoids the model confusing summer/winter baselines.
    print("  [9/9] Summer expanding-window profile...")

    is_summer_arr = np.isin(tm.values, list(SUMMER_MONTHS)).astype(int)
    prof_summer   = np.full(len(df), np.nan)

    for i in range(336, len(df)):
        mask = (
            (tgt_dow_arr[:i]  == tgt_dow_arr[i])  &
            (tgt_hour_arr[:i] == tgt_hour_arr[i]) &
            (is_work_arr[:i]  == is_work_arr[i])  &
            (is_summer_arr[:i] == is_summer_arr[i])
        )
        vals = energy_arr[:i][mask]
        if len(vals) >= 3:
            prof_summer[i] = np.median(vals[-10:])

    F["profile_median_summer"] = prof_summer

    # ── ASSEMBLE ─────────────────────────────────────────────
    df_feat = pd.DataFrame(F, index=df.index)

    df_feat["target_24h"] = e.shift(-HORIZON)

    df_feat.insert(0, "datetime", df["datetime"])
    if "sn" in df.columns:
        df_feat.insert(0, "sn", df["sn"])

    return df_feat


# ─────────────────────────────────────────────────────────────
# 4. CLEAN & VALIDATE
# ─────────────────────────────────────────────────────────────
LEAK_COLS = {
    "target_24h",
    "prof_median", "prof_std", "z_score", "regime_scale",
}

def clean_and_validate(df_feat: pd.DataFrame) -> pd.DataFrame:
    before = len(df_feat)

    df_feat = df_feat.dropna(subset=["target_24h"])
    df_feat = df_feat.dropna(subset=["profile_median"])

    df_feat = df_feat.replace([np.inf, -np.inf], np.nan)

    feat_cols = [c for c in df_feat.columns
                 if c not in {"datetime","sn","target_24h"} and df_feat[c].dtype != object]

    # Impute remaining NaNs using training split medians to avoid data leakage
    split = int(len(df_feat) * TRAIN_RATIO)
    for c in feat_cols:
        if df_feat[c].isna().any():
            train_median = df_feat[c].iloc[:split].median()
            df_feat[c] = df_feat[c].fillna(train_median)

    print(f"\n  Rows: {before:,} → {len(df_feat):,} (dropped {before-len(df_feat):,})")

    print("\n  [Leak check] Verifying no future energy in feature columns...")
    for col in feat_cols:
        if col in LEAK_COLS:
            print(f"    ⚠️  LEAK COL DETECTED: {col} — removing")
            df_feat = df_feat.drop(columns=[col])
        if col in df_feat.columns:  # Check if column still exists
            corr = df_feat[col].corr(df_feat["target_24h"])
            if abs(corr) > 0.99:
                print(f"    ⚠️  Suspiciously high corr ({corr:.4f}): {col}")

    print("  [Leak check] Done.")
    return df_feat


# ─────────────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  FEATURE ENGINEERING — 24h Building Energy Forecast")
    print("  Tunisia meter dataset")
    print("=" * 65)

    print(f"\n[1] Loading energy data: {INPUT_CSV.name}")
    df = pd.read_csv(INPUT_CSV, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df = df.drop(columns=["hour","month","is_work_hours"], errors="ignore")
    print(f"    {len(df):,} rows | {df['datetime'].min().date()} → {df['datetime'].max().date()}")

    print("\n[2] Weather data...")
    start_str = (df["datetime"].min() - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    end_str   = (df["datetime"].max() + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    wdf = download_weather(start_str, end_str)
    fc_wdf = download_weather_forecast_24h(start_str)

    print("\n[3] Merging energy + weather...")
    df = df.merge(wdf, on="datetime", how="left")
    # Forward-fill missing weather observations
    for col in wdf.columns[1:]:
        df[col] = df[col].ffill().bfill()
    missing_pct = df[wdf.columns[1:]].isna().mean().max() * 100
    print(f"    Max weather NaN after fill: {missing_pct:.2f}%")
    print(f"    Merged shape: {df.shape}")

    print("\n[4] Building features...")
    df_feat = build_features(df, fc_wdf)

    print("\n[5] Cleaning & validating...")
    df_feat = clean_and_validate(df_feat)

    feat_cols = [c for c in df_feat.columns
                 if c not in {"datetime","sn","target_24h"} and df_feat[c].dtype != object]
    print(f"\n{'='*65}")
    print(f"  FEATURE ENGINEERING COMPLETE")
    print(f"{'='*65}")
    print(f"  Output rows     : {len(df_feat):,}")
    print(f"  Feature columns : {len(feat_cols)}")
    print(f"  Date range      : {df_feat['datetime'].min().date()} → {df_feat['datetime'].max().date()}")
    print(f"  Target stats    : mean={df_feat['target_24h'].mean():.3f}  "
          f"std={df_feat['target_24h'].std():.3f}  "
          f"max={df_feat['target_24h'].max():.3f}")

    print("\n  Feature groups:")
    groups = {
        "Autoregressive lags":      [c for c in feat_cols if c.startswith("lag_")],
        "Rolling statistics":       [c for c in feat_cols if "roll_" in c or "ewm_" in c or "velocity" in c],
        "Calendar cyclical":        [c for c in feat_cols if any(x in c for x in ["sin","cos","t_hour","t_dow","t_month","t_doy"])],
        "Calendar flags":           [c for c in feat_cols if any(x in c for x in ["is_","bridge_"])],
        "Weather observed":         [c for c in feat_cols if any(x in c for x in ["_obs","cdd_","hdd_","consec","solar","humid","dew","temp_ema","temp_max","temp_min","temp_acc","temp_diurnal","heat_index","above_","temp_sq","daily_","night_","day_night"]) and "24h" not in c],
        "Weather forecast (t+24)":  [c for c in feat_cols if "24h" in c and "lag" not in c and "roll" not in c and "profile" not in c],
        "Interactions":             [c for c in feat_cols if "_x_" in c or "momentum" in c or "hour_x" in c or "dow_x" in c],
        "Profile":                  [c for c in feat_cols if "profile" in c],
    }
    for grp, cols in groups.items():
        if cols:
            print(f"    {grp:<30} {len(cols):>3} features")

    print(f"\n[6] Saving → {OUTPUT_CSV.name}")
    df_feat.to_csv(OUTPUT_CSV, index=False)
    print(f"    Saved: {df_feat.shape[0]:,} rows × {df_feat.shape[1]} columns")

    return df_feat


if __name__ == "__main__":
    main()