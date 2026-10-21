"""
Forecast Service — generates 24h-ahead consumption forecasts for energy and water.
Redis-cached (1h) and persisted to MySQL.
"""
from __future__ import annotations

import logging
import math
import os
import sys
from datetime import datetime, timedelta

import pandas as pd

from core.redis_client import cache_get, cache_set
from services.db_service import get_daily_consumption, save_forecasts
from services.model_service import ModelStore
from services.weather_service import get_weather

logger = logging.getLogger(__name__)

# Pinned reference date for demo consistency
DEMO_PIN_DATE = "2026-08-20"


_FORECAST_TTL = 3600  # 1 h
_LAG_CONTEXT_HOURS = 336  # 14 days

_BUILD_FEATURES = None


def _load_build_features():
    """Dynamically import the training feature engineering pipeline."""
    global _BUILD_FEATURES
    if _BUILD_FEATURES is not None:
        return _BUILD_FEATURES
    _env = os.getenv("ML_MODELS_DIR")
    _root = (os.path.abspath(_env) if _env
             else os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    energy_path = os.path.join(_root, "energy_forecasting", "features_engineering")
    if energy_path not in sys.path:
        sys.path.insert(0, energy_path)
    sys.modules.pop("config", None)
    sys.modules.pop("features_engineering", None)
    from features_engineering import build_features
    _BUILD_FEATURES = build_features
    return _BUILD_FEATURES


# ─────────────────────────────────────────────────────────────────────────────
#  door_open_prob lookup (water) — moved here verbatim from main.py
# ─────────────────────────────────────────────────────────────────────────────
async def get_door_open_prob_lookup(site_id: int) -> dict:
    """
    Empirical P(nonzero consumption) per (day_of_week, hour), replicating how
    water_forecasting/features_engineering/generate_features.py builds the
    door_open_prob feature at training time — computed live from DB actuals
    since the saved model doesn't ship this lookup table itself. Cached 12h
    since the pattern changes slowly.
    """
    cache_key = f"door_open_prob:{site_id}"
    cached = await cache_get(cache_key)
    if cached:
        return cached

    records = await get_daily_consumption(site_id, days=180)
    if not records:
        return {}

    df = pd.DataFrame(records)
    df["datetime"] = pd.to_datetime(df["date"], unit="ms")
    df["dow"] = df["datetime"].dt.dayofweek
    df["hour"] = df["datetime"].dt.hour
    df["nonzero"] = (df["total_consumption"].fillna(0) > 0).astype(float)

    grouped = df.groupby(["dow", "hour"])["nonzero"].mean()
    lookup = {f"{dow}_{hour}": float(v) for (dow, hour), v in grouped.items()}
    lookup["_mean"] = float(df["nonzero"].mean())

    await cache_set(cache_key, lookup, 43200)  # 12h
    return lookup


# ─────────────────────────────────────────────────────────────────────────────
#  Per-domain forecast computation
#  `cleaned` is the ETL-cleaned hourly series (list of {date ms, local_date,
#  total_consumption}), chronological — the SAME step-3 cleaned data the
#  models were trained on. See compute_forecast().
# ─────────────────────────────────────────────────────────────────────────────
async def _forecast_energy(cleaned: list, now: datetime) -> tuple[list, str]:
    """24 × t+24 predictions (rows T-23..T each predict their own +24h), using
    the REAL training feature pipeline so the feature set always matches the
    deployed model, fed the ETL-cleaned series (matches training input)."""
    observed_df, forecast_df = await get_weather(days=32)

    records = [
        {"datetime": datetime.utcfromtimestamp(r["date"] / 1000.0),
         "energy_kwh": r["total_consumption"] or 0.0}
        for r in cleaned
    ]
    if not records:
        now_dt = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        records = [{"datetime": now_dt - timedelta(hours=i), "energy_kwh": 0.5}
                   for i in range(_LAG_CONTEXT_HOURS, -1, -1)]
    while len(records) < _LAG_CONTEXT_HOURS:
        records.insert(0, {"datetime": records[0]["datetime"] - timedelta(hours=1),
                           "energy_kwh": records[0]["energy_kwh"]})

    df = pd.DataFrame(records)
    if not observed_df.empty:
        df = df.merge(observed_df, on="datetime", how="left")
        for col in observed_df.columns:
            if col != "datetime":
                df[col] = df[col].ffill().bfill()
    else:
        for col in ["temperature", "humidity", "cloud_cover", "solar_radiation", "dew_point"]:
            df[col] = 0.0

    df_feat = _load_build_features()(df, forecast_df)
    predict_df = df_feat.iloc[-24:].copy().fillna(0)

    result = ModelStore.predict_energy(predict_df)
    preds = result["predictions"] if result["predictions"] else [0.0] * 24

    forecasts = []
    for h in range(24):
        hour_dt = now + timedelta(hours=h + 1)
        val = preds[h] if h < len(preds) else 0.0
        forecasts.append({"date": int(hour_dt.timestamp() * 1000),
                          "predicted_value": round(float(val), 3)})
    return forecasts, result.get("model_used", "unknown")


async def _forecast_water(site_id: int, now: datetime, series_l: list) -> tuple[list, str]:
    """Recursive 24-step t+1 rollout on the ETL-cleaned LITRES series. The
    model is single-step, so each prediction is fed back into the next step's
    lag/rolling/ewma features. Everything here is litres — the DB, the model's
    training data (BDG2) and the output all agree, no conversions."""
    door_prob_lookup = await get_door_open_prob_lookup(site_id)
    door_prob_mean = door_prob_lookup.get("_mean", 0.5)

    vals = list(series_l)  # litres, chronological (oldest→newest)
    while len(vals) < _LAG_CONTEXT_HOURS:  # need at least lag_168h
        vals.insert(0, vals[0] if vals else 0.0)

    EWMA_SPAN = 6
    ewma_alpha = 2.0 / (EWMA_SPAN + 1)
    ewma_state = vals[0]
    for v in vals[1:]:
        ewma_state = ewma_alpha * v + (1 - ewma_alpha) * ewma_state

    forecasts = []
    model_used = "unknown"
    for h in range(24):
        hour_dt = now + timedelta(hours=h + 1)

        lag_1h, lag_2h, lag_3h = vals[-1], vals[-2], vals[-3]
        lag_24h, lag_25h, lag_168h = vals[-24], vals[-25], vals[-168]

        row = {
            "door_open_prob": door_prob_lookup.get(f"{hour_dt.weekday()}_{hour_dt.hour}", door_prob_mean),
            "lag_168h": lag_168h,
            "hour": hour_dt.hour,
            "lag_1h": lag_1h,
            "rolling_mean_3h": sum(vals[-3:]) / 3,
            "is_working_hour": int(hour_dt.weekday() < 5 and 8 <= hour_dt.hour <= 18),
            "hour_cos": math.cos(2 * math.pi * hour_dt.hour / 24),
            "ewma_6h": ewma_state,
            "lag_2h": lag_2h,
            "lag_24h": lag_24h,
            "rolling_mean_6h": sum(vals[-6:]) / 6,
            "diff_1h": lag_1h - lag_2h,
            "diff_24h": lag_1h - lag_25h,
            "lag_3h": lag_3h,
            "hour_sin": math.sin(2 * math.pi * hour_dt.hour / 24),
        }

        result = ModelStore.predict_water(pd.DataFrame([row]))
        if not result["predictions"]:
            raise ValueError("Water ML model failed to predict — triggering fallback.")
        model_used = result.get("model_used", "unknown")

        val = max(0.0, float(result["predictions"][0]))  # litres/h
        vals.append(val)
        ewma_state = ewma_alpha * val + (1 - ewma_alpha) * ewma_state

        forecasts.append({"date": int(hour_dt.timestamp() * 1000),
                          "predicted_value": round(float(val), 3)})
    return forecasts, model_used


def _naive_baseline(series: list) -> list:
    """Persistence fallback: repeat the last known 24h average, shaped by a
    daily sine, for 24 h. Used only when live inference raises. `series` is
    the chronological consumption values (native unit)."""
    tail = series[-24:]
    base = sum(tail) / max(len(tail), 1)
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    out = []
    for h in range(24):
        hour_dt = now + timedelta(hours=h + 1)
        factor = math.sin(((hour_dt.hour - 6) / 24) * math.pi * 2) * 0.3 + 1
        out.append({"date": int(hour_dt.timestamp() * 1000),
                    "predicted_value": round(base * factor, 3)})
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Public entry point
# ─────────────────────────────────────────────────────────────────────────────
async def compute_forecast(domain: str, cleaned: list | None = None, cap_date: str | None = None) -> dict:
    """Generate 24h-ahead consumption forecast and historical actuals."""
    site_id = 2 if domain == "water" else 1
    cache_key = f"forecast_v3:{domain}:{cap_date or 'live'}"

    cached = await cache_get(cache_key)
    if cached and cached.get("forecasts"):
        cached.setdefault("model_used", "cached")
        return cached

    # Cleaned hourly series from data_agent
    if not cleaned:
        cleaned = []
        fetch_cap_date = DEMO_PIN_DATE if cap_date is None else cap_date
        try:
            from agents.data_agent import data_agent  # local: avoid import cycle at module load
            etl_state = await data_agent(
                {"domains": [domain], "days": 30, "needs_forecast": True, "needs_anomaly": False,
                 "demo_cap_date": fetch_cap_date}
            )
            cleaned = etl_state["consumption_records"].get(domain, []) or []
        except Exception as e:
            logger.warning("[forecast_service] %s forecast-ETL failed (%s) — raw fallback", domain, e)

    if not cleaned:
        raw = await get_daily_consumption(site_id, 30, cap_date=cap_date)  # newest-first
        cleaned = [{"date": r["date"], "local_date": None,
                    "total_consumption": r["total_consumption"]}
                   for r in reversed(raw)]

    cleaned = sorted(cleaned, key=lambda r: r["date"])  # chronological; copy (may be caller's list)
    series = [(r["total_consumption"] or 0.0) for r in cleaned]

    if cleaned:
        now = datetime.utcfromtimestamp(cleaned[-1]["date"] / 1000.0).replace(minute=0, second=0, microsecond=0)
    else:
        now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)

    try:
        if domain == "energy":
            forecasts, model_used = await _forecast_energy(cleaned, now)
        else:
            forecasts, model_used = await _forecast_water(site_id, now, series)
    except Exception as e:
        logger.warning("[forecast_service] %s live forecast failed (%s) — naive baseline", domain, e)
        forecasts = _naive_baseline(series or [0.5])
        model_used = "naive-baseline"

    response = {"actuals": cleaned, "forecasts": forecasts, "model_used": model_used}

    # Persistence + cache are best-effort — a DB or Redis hiccup must not
    # discard 24 already-computed predictions.
    try:
        await save_forecasts(site_id, [f["predicted_value"] for f in forecasts], model_used, domain)
    except Exception as e:
        logger.warning("[forecast_service] save_forecasts(%s) failed: %s", domain, e)
    try:
        await cache_set(cache_key, response, _FORECAST_TTL)
    except Exception as e:
        logger.warning("[forecast_service] cache_set(%s) failed: %s", domain, e)

    return response
