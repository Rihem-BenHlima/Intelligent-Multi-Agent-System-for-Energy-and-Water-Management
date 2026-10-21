"""
Fetches live observed and 24h forecast weather data from the Open-Meteo API.
Results are cached in Redis.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

from core.redis_client import cache_get, cache_set

logger = logging.getLogger(__name__)

# Coordinates for Tunis, Tunisia
_LATITUDE  = 36.8065
_LONGITUDE = 10.1815
_TIMEZONE  = "Africa/Tunis"

_ARCHIVE_URL      = "https://archive-api.open-meteo.com/v1/archive"
_PREV_RUNS_URL    = "https://previous-runs-api.open-meteo.com/v1/forecast"

_CACHE_TTL = 3600          # 1 hour
_TIMEOUT   = 30            # seconds per API request


# ── Observed Weather ──────────────────────────────────────────────────────────

async def get_observed_weather(days: int = 8) -> pd.DataFrame:
    """
    Fetch hourly observed weather for the last `days` days.
    Returns DataFrame with temperature, humidity, dew_point, solar_radiation, etc.
    """
    cache_key = f"weather:observed:{days}"
    cached = await cache_get(cache_key)
    if cached:
        logger.info("[WeatherService] Observed weather served from Redis cache.")
        df = pd.DataFrame(cached)
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df

    now        = datetime.now(timezone.utc)
    start_date = (now - timedelta(days=days)).strftime("%Y-%m-%d")
    end_date   = now.strftime("%Y-%m-%d")

    params = {
        "latitude":   _LATITUDE,
        "longitude":  _LONGITUDE,
        "start_date": start_date,
        "end_date":   end_date,
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
        "timezone": _TIMEZONE,
    }

    try:
        logger.info("[WeatherService] Fetching observed weather from Open-Meteo Archive API...")
        resp = requests.get(_ARCHIVE_URL, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()["hourly"]

        df = pd.DataFrame({
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

        # Remove timezone info if present (matches training behaviour)
        if df["datetime"].dt.tz is not None:
            df["datetime"] = df["datetime"].dt.tz_localize(None)

        df = (df.drop_duplicates("datetime")
                .sort_values("datetime")
                .reset_index(drop=True))

        # Fill any gaps
        for col in df.columns[1:]:
            df[col] = df[col].ffill().bfill()

        logger.info("[WeatherService] Observed weather fetched: %d rows (%s → %s)",
                    len(df), start_date, end_date)

        # Cache as list of dicts (JSON-serialisable)
        payload = df.copy()
        payload["datetime"] = payload["datetime"].astype(str)
        await cache_set(cache_key, payload.to_dict(orient="records"), ttl=_CACHE_TTL)

        return df

    except Exception as exc:
        logger.error("[WeatherService] Observed weather fetch failed: %s", exc)
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
#  24H-AHEAD FORECAST WEATHER
# ── 24h-Ahead Forecast Weather ────────────────────────────────────────────────

async def get_forecast_weather_24h(days: int = 8) -> pd.DataFrame:
    """
    Fetch 24h-ahead weather forecast for the last `days` days from Open-Meteo Previous Runs API.
    Returns DataFrame with temp_24h, humidity_24h, cloud_24h, solar_rad_24h.
    """
    cache_key = f"weather:forecast24h:{days}"
    cached = await cache_get(cache_key)
    if cached:
        logger.info("[WeatherService] 24h forecast weather served from Redis cache.")
        df = pd.DataFrame(cached)
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df

    variables = [
        "temperature_2m",
        "relative_humidity_2m",
        "cloud_cover",
        "shortwave_radiation",
    ]

    params = {
        "latitude":       _LATITUDE,
        "longitude":      _LONGITUDE,
        "hourly":         variables + [f"{v}_previous_day1" for v in variables],
        "past_days":      days,
        "forecast_days":  1,
        "timezone":       _TIMEZONE,
    }

    try:
        logger.info("[WeatherService] Fetching 24h-ahead forecast from Open-Meteo Previous Runs API...")
        resp = requests.get(_PREV_RUNS_URL, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()["hourly"]

        df = pd.DataFrame({
            "datetime":      pd.to_datetime(data["time"]),
            "temp_24h":      data["temperature_2m_previous_day1"],
            "humidity_24h":  data["relative_humidity_2m_previous_day1"],
            "cloud_24h":     data["cloud_cover_previous_day1"],
            "solar_rad_24h": data["shortwave_radiation_previous_day1"],
        })

        if df["datetime"].dt.tz is not None:
            df["datetime"] = df["datetime"].dt.tz_localize(None)

        df = (df.dropna(subset=["temp_24h"])
                .drop_duplicates("datetime")
                .sort_values("datetime")
                .reset_index(drop=True))

        logger.info("[WeatherService] 24h forecast weather fetched: %d rows", len(df))

        payload = df.copy()
        payload["datetime"] = payload["datetime"].astype(str)
        await cache_set(cache_key, payload.to_dict(orient="records"), ttl=_CACHE_TTL)

        return df

    except Exception as exc:
        logger.error("[WeatherService] 24h forecast weather fetch failed: %s", exc)
        return pd.DataFrame()


# ── Forward-Looking Forecast (Dashboard display) ──────────────────────────────

_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_CURRENT_CACHE_TTL = 900   # 15 minutes


async def get_current_weather() -> dict | None:
    """
    Fetch current live weather conditions via Open-Meteo Forecast API.
    Returns dict: {datetime, temperature, humidity, cloud_cover} or None.
    """
    cache_key = "weather:current"
    cached = await cache_get(cache_key)
    if cached:
        logger.info("[WeatherService] Current weather served from Redis cache.")
        return cached

    params = {
        "latitude":  _LATITUDE,
        "longitude": _LONGITUDE,
        "current":   "temperature_2m,relative_humidity_2m,cloud_cover",
        "timezone":  _TIMEZONE,
    }

    try:
        logger.info("[WeatherService] Fetching live current weather from Open-Meteo Forecast API...")
        resp = requests.get(_FORECAST_URL, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        current = resp.json()["current"]

        result = {
            "datetime":    current["time"],
            "temperature": current["temperature_2m"],
            "humidity":    current["relative_humidity_2m"],
            "cloud_cover": current["cloud_cover"],
        }

        logger.info("[WeatherService] Current weather fetched: %.1f°C at %s",
                    result["temperature"], result["datetime"])

        await cache_set(cache_key, result, ttl=_CURRENT_CACHE_TTL)
        return result

    except Exception as exc:
        logger.error("[WeatherService] Current weather fetch failed: %s", exc)
        return None


def seconds_until_midnight_utc() -> int:
    """Return seconds remaining until the next UTC midnight."""
    now = datetime.utcnow()
    next_midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(1, int((next_midnight - now).total_seconds()))


async def get_tomorrow_forecast() -> pd.DataFrame:
    """
    Fetch forward-looking hourly weather forecast for tomorrow.
    Returns DataFrame with datetime, temperature, humidity, cloud_cover.
    """
    cache_key = "weather:forecast:tomorrow"
    cached = await cache_get(cache_key)
    if cached:
        logger.info("[WeatherService] Tomorrow forecast served from Redis cache.")
        df = pd.DataFrame(cached)
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df

    params = {
        "latitude":      _LATITUDE,
        "longitude":     _LONGITUDE,
        "hourly":        ["temperature_2m", "relative_humidity_2m", "cloud_cover"],
        "forecast_days": 2,
        "timezone":      _TIMEZONE,
    }

    try:
        logger.info("[WeatherService] Fetching tomorrow's forecast from Open-Meteo Forecast API...")
        resp = requests.get(_FORECAST_URL, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()["hourly"]

        df = pd.DataFrame({
            "datetime":    pd.to_datetime(data["time"]),
            "temperature": data["temperature_2m"],
            "humidity":    data["relative_humidity_2m"],
            "cloud_cover": data["cloud_cover"],
        })
        df = df.dropna(subset=["temperature"]).sort_values("datetime").reset_index(drop=True)

        logger.info("[WeatherService] Tomorrow forecast fetched: %d rows", len(df))

        payload = df.copy()
        payload["datetime"] = payload["datetime"].astype(str)
        await cache_set(cache_key, payload.to_dict(orient="records"), ttl=seconds_until_midnight_utc())

        return df

    except Exception as exc:
        logger.error("[WeatherService] Tomorrow forecast fetch failed: %s", exc)
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
#  CONVENIENCE — fetch both together
# ═══════════════════════════════════════════════════════════════════════════════

async def get_weather(days: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch both observed and 24h-ahead forecast weather concurrently.
    Returns (observed_df, forecast_df).
    """
    import asyncio
    observed_task  = asyncio.create_task(get_observed_weather(days))
    forecast_task  = asyncio.create_task(get_forecast_weather_24h(days))
    observed_df, forecast_df = await asyncio.gather(observed_task, forecast_task)
    return observed_df, forecast_df
