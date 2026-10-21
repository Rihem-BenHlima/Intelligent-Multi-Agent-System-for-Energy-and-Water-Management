"""
LangGraph node — Anomaly Detection Agent.
Detects energy and water consumption anomalies using domain-specific LSTM Autoencoders.
"""

import logging
from datetime import datetime, timezone

from services.model_service import ModelStore, ANOMALY_DETECTION_DAYS
from services.db_service import save_anomalies, get_recent_anomalies, describe_anomaly_event
from core.redis_client import cache_get, cache_set

logger = logging.getLogger(__name__)

_ANOMALY_TIMELINE_TTL = 3600  # 1h


def _iso(ms) -> str:
    """Normalise epoch-ms timestamp to ISO string."""
    try:
        return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).replace(tzinfo=None).isoformat()
    except (TypeError, ValueError):
        return str(ms)


async def anomaly_agent(state: dict) -> dict:
    domains: list[str] = state.get("domains") or ["energy"]

    agent_log: list[str] = list(state.get("agent_log", []))
    anomalies_dict = {}
    threshold_dict = {}

    for domain in domains:
        site_id = 2 if domain == "water" else 1

        demo_cap_date = state.get("demo_cap_date")
        timeline_cache_key = f"anomaly_timeline:{domain}:{demo_cap_date or 'live'}"
        cached_timeline = await cache_get(timeline_cache_key)
        if cached_timeline:
            persisted = await get_recent_anomalies(device_id=site_id, domain=domain, limit=50)
            anomalies_dict[domain] = [
                {"start": _iso(r.get("start_date")), "end": _iso(r.get("end_date")),
                 "description": r.get("description", "")}
                for r in persisted
            ]
            threshold_dict[domain] = "cached"
            logger.info("[AnomalyAgent] Reusing cached detection for domain=%s (%d persisted events)",
                        domain, len(persisted))
            agent_log.append(f"AnomalyAgent: {domain} anomalies reused from cache ({len(persisted)} events)")
            continue

        from agents.data_agent import data_agent
        domain_state = await data_agent(
            {"domains": [domain], "days": ANOMALY_DETECTION_DAYS,
             "needs_forecast": False, "needs_anomaly": True,
             "demo_cap_date": demo_cap_date}
        )
        data = domain_state["raw_consumption_data"].get(domain, [])
        timestamps = domain_state["raw_consumption_timestamps"].get(domain, [])

        if not data or len(data) != len(timestamps):
            logger.warning("AnomalyAgent: no/misaligned raw_consumption_data for %s — skipping.", domain)
            continue

        if domain == "water":
            result = ModelStore.detect_water_anomalies(data, timestamps)
        else:
            result = ModelStore.detect_energy_anomalies(data, timestamps)

        events = result["events"]
        anomalies_dict[domain] = [
            {"start": ev["start"], "end": ev["end"],
             "description": describe_anomaly_event(ev, domain)}
            for ev in events
        ]
        threshold_dict[domain] = result.get("method")

        anomaly_count = len(events)
        logger.info(
            "[AnomalyAgent] domain=%s  events=%d  windows=%d  method=%s",
            domain, anomaly_count, result.get("n_windows", 0), result.get("method"),
        )

        save_success = await save_anomalies(device_id=site_id, events=events, domain=domain)

        timeline = result.get("timeline") or []
        if timeline:
            try:
                await cache_set(timeline_cache_key, timeline[-720:], _ANOMALY_TIMELINE_TTL)
            except Exception as exc:
                logger.warning("[AnomalyAgent] anomaly_timeline cache_set failed for %s: %s", domain, exc)

        log_msg = f"AnomalyAgent: detected {anomaly_count} {domain} anomaly event(s)"
        if save_success:
            log_msg += " (saved to DB)"
        else:
            log_msg += " (failed to save to DB)"

        agent_log.append(log_msg)

    return {
        **state,
        "anomalies":         anomalies_dict,
        "anomaly_threshold": threshold_dict,
        "agent_log":         agent_log,
    }
