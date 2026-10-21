"""
LangGraph node — Recommendation Agent.
Generates domain-specific energy and water conservation tips.
"""

import json
import logging
import re
from services.llm_service import LLMService
from core.redis_client import cache_get, cache_set

_RECS_CACHE_TTL = 3600  # 1h

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert in energy and water efficiency. "
    "Based on the provided consumption data, forecasted values, and detected anomalies, "
    "generate 3-5 practical usage and saving TIPS — not analysis or commentary. "
    "The FIRST WORD of every tip must be a verb in imperative mood "
    "('Turn off...', 'Shift...', 'Install...', 'Set...', 'Unplug...', "
    "'Inspect...', 'Schedule...'). "
    "NEVER start a tip with 'The', 'Because', 'Since', 'Given', 'Due to', "
    "'Your', 'With', or any other word that begins a description instead "
    "of a command — the pattern/number belongs at the END of the "
    "sentence as a short justification, not the beginning. "
    "Example of the required shape: \"Unplug standby electronics overnight "
    "— consumption stays flat at 0.52 kWh with no active use.\" "
    "Example of what NOT to do: \"The flat consumption of 0.52 kWh "
    "indicates a standby load; unplug electronics.\" "
    "Focus strictly on the numerical patterns and anomalies provided. "
    "Do not invent facts. "
    "Adapt your advice to whichever domains (energy and/or water) are present in the data. "
    "Electricity quantities are in kWh; water quantities are in litres (L) — never cubic metres. "
    "Respond with ONLY a JSON array of 3-5 strings — one self-contained "
    "tip per string. No markdown, no nested bullets, no code fences, no "
    "extra keys — just the raw JSON array."
)


def _flatten_text(val) -> str:
    """Flatten message content to string."""
    if isinstance(val, list):
        return " ".join([b.get("text", "") if isinstance(b, dict) else str(b) for b in val])
    return str(val) if val else ""


def _parse_tips(raw) -> list[str]:
    """Parse model JSON-array response into clean tip strings."""
    cleaned = _flatten_text(raw).strip()
    fence_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list) and all(isinstance(t, str) for t in parsed):
            return [t.strip() for t in parsed if t.strip()]
    except json.JSONDecodeError:
        pass
    logger.warning("[RecommendationAgent] Model did not return valid JSON tips, falling back to raw text")
    return [cleaned] if cleaned else []


async def recommendation_agent(state: dict) -> dict:
    """LangGraph node — Recommendation Agent."""
    domains          = state.get("domains") or ["energy"]
    anomalies        = state.get("anomalies") or {}
    forecasts        = state.get("forecasts") or {}
    consumption_data = state.get("consumption_data") or {}

    # Check cache for single-domain requests
    if len(domains) == 1:
        cached = await cache_get(f"recommendations:{domains[0]}")
        if cached and cached.get("recommendations"):
            logger.info("[RecommendationAgent] Reusing cached recommendations for domain=%s", domains[0])
            tips = cached["recommendations"]
            agent_log = list(state.get("agent_log", []))
            agent_log.append({
                "agent": "recommendation_agent",
                "input": {"domains": domains, "source": "cache"},
                "output": tips,
            })
            return {**state, "recommendations": tips, "agent_log": agent_log}

    # Build domain-aware context
    sections = []
    for domain in domains:
        unit   = "kWh"  if domain == "energy" else "Litres"
        data   = consumption_data.get(domain, [])
        f_vals = forecasts.get(domain, [])
        a_vals = anomalies.get(domain, [])

        recent_data = data[-24:] if data else []

        section = (
            f"=== {domain.upper()} DOMAIN ===\n"
            f"Recent 24-period consumption : {recent_data} {unit}\n"
            f"Anomalies detected           : {len(a_vals)}\n"
            f"Anomaly details              : {a_vals if a_vals else 'None'}\n"
            f"24h-ahead forecast           : {f_vals if f_vals else 'N/A'} {unit}\n"
        )
        sections.append(section)

    context = "\n".join(sections)

    user_message = (
        f"Please provide practical saving recommendations for the following domain(s): {', '.join(domains)}.\n\n"
        f"Data context:\n{context}"
    )

    logger.info("[RecommendationAgent] Building recommendations for domains=%s", domains)

    response_text = await LLMService.chat(
        system_prompt=SYSTEM_PROMPT,
        user_message=user_message,
        fast=True,
    )

    tips = _parse_tips(response_text)
    if not tips:
        tips = ["No specific recommendations could be generated from the available data."]

    # Cache recommendations for single-domain responses
    if len(domains) == 1 and not (len(tips) == 1 and LLMService.is_failure(tips[0])):
        try:
            await cache_set(f"recommendations:{domains[0]}",
                            {"recommendations": tips, "domain": domains[0]},
                            _RECS_CACHE_TTL)
        except Exception as exc:
            logger.warning("[RecommendationAgent] cache_set failed for %s: %s", domains[0], exc)

    agent_log = list(state.get("agent_log", []))
    agent_log.append({
        "agent": "recommendation_agent",
        "input": {
            "domains":            domains,
            "anomalies_count":    {d: len(anomalies.get(d, []))        for d in domains},
            "forecasts_count":    {d: len(forecasts.get(d, []))        for d in domains},
            "recent_consumption": {d: len(consumption_data.get(d, [])) for d in domains},
        },
        "output": tips,
    })

    return {
        **state,
        "recommendations": tips,
        "agent_log":       agent_log,
    }
