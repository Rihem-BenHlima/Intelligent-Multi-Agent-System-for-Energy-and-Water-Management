"""
LangGraph node — Forecasting Agent.
Wraps services.forecast_service.compute_forecast() for energy and water.
"""

from __future__ import annotations

import logging

from services.forecast_service import compute_forecast

logger = logging.getLogger(__name__)


async def forecast_agent(state: dict) -> dict:
    """LangGraph node — Forecasting Agent."""
    domains: list[str] = state.get("domains") or ["energy"]
    agent_log: list[str] = list(state.get("agent_log") or [])
    cleaned_records: dict = state.get("consumption_records") or {}

    forecasts_dict: dict = {}
    model_used_dict: dict = {}

    for domain in domains:
        try:
            result = await compute_forecast(domain, cleaned=cleaned_records.get(domain) or None)
            values = [f["predicted_value"] for f in result.get("forecasts", [])]
            model_used = result.get("model_used", "unknown")
        except Exception as exc:
            logger.error("[ForecastAgent] %s forecast failed: %s", domain, exc, exc_info=True)
            agent_log.append(f"ForecastAgent: {domain} forecast failed ({exc})")
            continue

        if not values:
            agent_log.append(f"ForecastAgent: {domain} forecast unavailable")
            continue

        forecasts_dict[domain] = values
        model_used_dict[domain] = model_used
        logger.info("[ForecastAgent] domain=%s  model=%s  n=%d", domain, model_used, len(values))
        agent_log.append(f"ForecastAgent: {domain} forecast ready ({len(values)} pts via {model_used})")

    return {
        **state,
        "forecasts": forecasts_dict,
        "model_used": model_used_dict,
        "agent_log": agent_log,
    }
