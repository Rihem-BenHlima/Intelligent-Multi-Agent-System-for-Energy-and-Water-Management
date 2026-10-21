"""
Data Agent — retrieves consumption records from MySQL and executes
required domain-specific ETL pipelines (forecast ETL and anomaly ETL).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
import pandas as pd

from services.db_service import get_daily_consumption

from etl import (
    energy_forecast_etl,
    energy_anomaly_etl,
    water_forecast_etl,
    water_anomaly_etl,
)

logger = logging.getLogger(__name__)


def _infer_etl_needs(query: str) -> tuple[bool, bool]:
    """Infer required ETL pipelines (forecast_etl, anomaly_etl) from query keywords."""
    q = (query or "").lower()
    is_anomaly_kw = any(kw in q for kw in ("anomal", "detect"))
    is_report_kw  = any(kw in q for kw in ("report", "summary"))
    if is_report_kw:
        return True, True
    if any(kw in q for kw in ("predict", "forecast")):
        return True, is_anomaly_kw
    if is_anomaly_kw:
        return False, True
    if any(kw in q for kw in ("recommend", "tip", "saving")):
        return True, is_anomaly_kw
    if any(kw in q for kw in ("cost", "bill", "tariff", "price", "invoice", "millime",
                              "dinar", "expensive", "pay", "how much is", "how much are")):
        return True, False
    return True, False


async def _run_etl_with_retry(
    etl_fn: Callable[[pd.DataFrame], pd.DataFrame],
    df_mysql: pd.DataFrame,
    *, label: str, retries: int = 2, delay: float = 0.5,
) -> Optional[pd.DataFrame]:
    """Execute an ETL function with retry on failure."""
    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 2):
        try:
            return etl_fn(df_mysql)
        except Exception as e:
            last_exc = e
            logger.warning("[DataAgent] %s attempt %d/%d failed: %s",
                           label, attempt, retries + 1, e)
            if attempt <= retries:
                await asyncio.sleep(delay)
    logger.error("[DataAgent] %s gave up after %d attempts: %s",
                 label, retries + 1, last_exc, exc_info=True)
    return None


def _daily_summary_from_df(df: pd.DataFrame, value_col: str, dt_col: str = "datetime") -> Dict[str, Any]:
    """Compute aggregate consumption statistics grouped by calendar day."""
    if df is None or df.empty:
        return {"total": 0.0, "avg_daily": 0.0, "max_daily": 0.0, "record_count": 0}
    daily = df.groupby(df[dt_col].dt.date)[value_col].sum()
    return {
        "total": float(daily.sum()),
        "avg_daily": float(daily.mean()),
        "max_daily": float(daily.max()),
        "record_count": int(len(daily)),
    }


async def data_agent(state: dict) -> dict:
    """LangGraph node — Data Agent: fetches raw consumption data and runs ETL pipelines."""
    domains: List[str] = state.get("domains") or ["energy"]
    days: int = state.get("days") or 7
    demo_cap_date: Optional[str] = state.get("demo_cap_date")

    default_need_forecast, default_need_anomaly = _infer_etl_needs(state.get("user_query", ""))
    need_forecast: bool = state.get("needs_forecast", default_need_forecast)
    need_anomaly: bool = state.get("needs_anomaly", default_need_anomaly)

    logger.info(
        "[DataAgent] Fetching consumption data for domains=%s, days=%s "
        "(forecast_etl=%s, anomaly_etl=%s)",
        domains, days, need_forecast, need_anomaly,
    )

    consumption_data = {}
    raw_consumption_data = {}
    raw_consumption_timestamps = {}
    consumption_records = {}
    consumption_summary = {}
    
    total_records = 0
    total_forecast_values = 0
    total_anomaly_values = 0

    for domain in domains:
        site_id = 2 if domain == "water" else 1

        try:
            records = await get_daily_consumption(site_id, days, cap_date=demo_cap_date)
            domain_records = [
                r if isinstance(r, dict) else {"date": getattr(r, "date", 0), "total_consumption": getattr(r, "total_consumption", 0.0)}
                for r in records
            ]
        except Exception as exc:
            logger.error("[DataAgent] get_daily_consumption failed for %s: %s", domain, exc, exc_info=True)
            domain_records = []

        domain_consumption = []
        domain_raw_consumption = []
        domain_raw_timestamps = []
        domain_summary = {"total": 0.0, "avg_daily": 0.0, "max_daily": 0.0, "record_count": 0}

        if domain_records:
            df_mysql = pd.DataFrame(domain_records)
            col_name = "consumption_l" if domain == "water" else "energy_kwh"

            # Forecast-ETL: cleaned/imputed series for forecasting and dashboards
            if need_forecast:
                forecast_fn = water_forecast_etl if domain == "water" else energy_forecast_etl
                df_forecast = await _run_etl_with_retry(
                    forecast_fn, df_mysql, label=f"Forecast-ETL[{domain}]")
                if df_forecast is not None:
                    domain_consumption = df_forecast[col_name].tolist()
                    domain_summary = _daily_summary_from_df(df_forecast, col_name)
                    domain_records = [
                        {
                            "date": int(row["datetime"].timestamp() * 1000),
                            "local_date": row["datetime"].date().isoformat(),
                            "total_consumption": float(row[col_name]),
                        }
                        for _, row in df_forecast.iterrows()
                    ]

            # Anomaly-ETL: raw spike-preserving series for anomaly detection
            if need_anomaly:
                anomaly_fn = water_anomaly_etl if domain == "water" else energy_anomaly_etl
                df_anomaly = await _run_etl_with_retry(
                    anomaly_fn, df_mysql, label=f"Anomaly-ETL[{domain}]")
                if df_anomaly is not None:
                    anomaly_col_name = "consumption_l" if domain == "water" else col_name
                    domain_raw_consumption = df_anomaly[anomaly_col_name].tolist()
                    domain_raw_timestamps = df_anomaly["datetime"].dt.strftime(
                        "%Y-%m-%dT%H:%M:%S").tolist()

        consumption_records[domain] = domain_records
        consumption_summary[domain] = domain_summary
        consumption_data[domain] = domain_consumption
        raw_consumption_data[domain] = domain_raw_consumption
        raw_consumption_timestamps[domain] = domain_raw_timestamps
        
        total_records += len(domain_records)
        total_forecast_values += len(domain_consumption)
        total_anomaly_values += len(domain_raw_consumption)

    logger.info(
        "[DataAgent] Retrieved %d records, %d forecast values, %d anomaly values total",
        total_records, total_forecast_values, total_anomaly_values
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    log_entry = (
        f"[{timestamp}] DataAgent — domains={domains}, days={days}, "
        f"records={total_records}, values={total_forecast_values}"
    )

    existing_log: List[str] = state.get("agent_log", [])

    return {
        **state,
        "consumption_data": consumption_data,
        "raw_consumption_data": raw_consumption_data,
        "raw_consumption_timestamps": raw_consumption_timestamps,
        "consumption_records": consumption_records,
        "consumption_summary": consumption_summary,
        "agent_log": existing_log + [log_entry],
    }
