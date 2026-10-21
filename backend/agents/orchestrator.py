"""
LangGraph orchestrator managing multi-agent routing and state transitions.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from typing_extensions import TypedDict

from agents.data_agent import data_agent
from agents.forecast_agent import forecast_agent
from agents.anomaly_agent import anomaly_agent
from agents.chat_agent import chat_agent
from agents.recommendation_agent import recommendation_agent
from agents.report_agent import report_agent
from core.redis_client import get_redis_checkpointer

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_query: str
    domains: List[str]
    days: int
    demo_cap_date: Optional[str]

    consumption_data: Dict[str, List[float]]
    raw_consumption_data: Dict[str, List[float]]
    raw_consumption_timestamps: Dict[str, List[str]]
    consumption_records: Dict[str, List[Dict]]
    consumption_summary: Dict[str, Dict]

    forecasts: Dict[str, List[float]]
    model_used: Dict[str, str]
    anomalies: Dict[str, List[Dict]]
    anomaly_threshold: Dict[str, str]

    recommendations: Any
    report: str

    next_agent: str
    agent_log: List[str]
    needs_forecast: bool
    needs_anomaly: bool
    turn_id: str


_PER_TURN_OUTPUT_KEYS = (
    "consumption_data", "raw_consumption_data", "raw_consumption_timestamps",
    "consumption_records", "consumption_summary",
    "forecasts", "model_used", "anomalies", "anomaly_threshold",
    "recommendations", "report",
)

_COST_KEYWORDS = (
    "cost", "bill", "tariff", "price", "invoice", "millime",
    "dinar", "expensive", "pay", "how much is", "how much are",
)

_CONSUMPTION_KEYWORDS = (
    "consumption", "consume", "usage", "used", "how much energy",
    "how much water", "how much electricity", "how much power",
    "how much did", "kwh", "kilowatt", "litre", "liter",
)

_QUERY_DATE_RE = re.compile(
    r"\b\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\b"
    r"|\b\d{4}-\d{1,2}-\d{1,2}\b"
    r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}\b"
    r"|\b\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b",
    re.IGNORECASE,
)

_COST_HISTORY_DAYS = 95
_FORECAST_HISTORY_DAYS = 30
_FORECAST_INTENTS = {"forecast", "report", "recommend"}


def _get_user_query(state: AgentState) -> str:
    """Resolve the canonical user query string."""
    query: str = state.get("user_query", "")
    if query:
        return query.lower()

    messages: List[BaseMessage] = state.get("messages", [])
    if messages:
        last = messages[-1]
        content = getattr(last, "content", "") or ""
        return str(content).lower()

    return ""


async def orchestrator_node(state: AgentState) -> AgentState:
    """Evaluate current state and route to the next agent."""
    query = _get_user_query(state)

    messages = state.get("messages", [])
    turn_id = next(
        (getattr(m, "id", None) for m in reversed(messages)
         if getattr(m, "type", "") == "human"),
        None,
    )
    new_turn = turn_id is not None and turn_id != state.get("turn_id")

    domains = list(state.get("domains") or [])
    if not domains:
        if "energy" in query or "electricity" in query or "power" in query:
            domains.append("energy")
        if "water" in query or "leak" in query:
            domains.append("water")
        if not domains:
            domains = ["energy", "water"]

    working_state: AgentState = {**state, "user_query": query, "domains": domains}

    if new_turn:
        for key in _PER_TURN_OUTPUT_KEYS:
            working_state[key] = None
        working_state["turn_id"] = turn_id
        _turn_intent = classify_intent(query)
        working_state["days"] = (
            _COST_HISTORY_DAYS if _turn_intent == "cost"
            else _FORECAST_HISTORY_DAYS if _turn_intent in _FORECAST_INTENTS
            else 7
        )
        logger.info("[Orchestrator] new turn (turn_id=%s) — cleared previous analytical outputs", turn_id)

    next_agent, needs_forecast, needs_anomaly = _decide_next_step(working_state)

    logger.info(
        "[Orchestrator] query=%r → domains=%s, next_agent=%s", query[:120], domains, next_agent
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    log_entry = (
        f"[{timestamp}] Orchestrator — query={query[:80]!r}, domains={domains}, routed_to={next_agent}"
    )
    existing_log: List[str] = state.get("agent_log", [])

    result: AgentState = {
        **working_state,
        "next_agent": next_agent,
        "agent_log": existing_log + [log_entry],
    }
    if needs_forecast is not None:
        result["needs_forecast"] = needs_forecast
    if needs_anomaly is not None:
        result["needs_anomaly"] = needs_anomaly
    return result


def route_from_orchestrator(state: AgentState) -> str:
    """Return next_agent decided by orchestrator."""
    return state["next_agent"]


def _has_possessive(query: str) -> bool:
    """Check for 'my' or 'our' to distinguish personal requests from generic definitions."""
    padded = f" {query} "
    return " my " in padded or " our " in padded


def classify_intent(query: str) -> str:
    """Classify user query intent."""
    is_informational = any(kw in query for kw in
        ("what is", "explain", "define", "tell me", "how does", "what are"))
    if is_informational and not _has_possessive(query):
        return "chat"
    if any(kw in query for kw in ("report", "summary")):
        return "report"
    if any(kw in query for kw in _COST_KEYWORDS):
        return "cost"
    if any(kw in query for kw in ("predict", "forecast")):
        return "forecast"
    if any(kw in query for kw in ("anomal", "detect")):
        return "anomaly"
    if any(kw in query for kw in ("recommend", "tip", "saving")):
        return "recommend"
    if any(kw in query for kw in _CONSUMPTION_KEYWORDS) or _QUERY_DATE_RE.search(query):
        return "consumption"
    return "chat"


def _decide_next_step(state: AgentState) -> tuple[str, Optional[bool], Optional[bool]]:
    """Determine the next agent execution step based on current state."""
    query = state.get("user_query", "")
    intent = classify_intent(query)

    # Stage 0: Conversational queries route directly to chat_agent
    if intent == "chat":
        messages = state.get("messages", [])
        last_msg = messages[-1] if messages else None
        chat_already_responded = last_msg is not None and getattr(last_msg, "type", "") == "ai"
        return (END if chat_already_responded else "chat_agent"), None, None

    # Stage 1: Data Agent must fetch/clean data before analytical tasks
    if state.get("consumption_data") is None and state.get("raw_consumption_data") is None:
        needs_forecast, _needs_anomaly = _etl_needs_for_query(query)
        return "data_agent", needs_forecast, False

    # Stage 2/3: ML models (forecast / anomaly)
    if intent not in ("consumption", "cost"):
        needs_forecast, needs_anomaly = _etl_needs_for_query(query)
        if needs_forecast and state.get("forecasts") is None:
            return "forecast_agent", None, None
        if needs_anomaly and state.get("anomalies") is None:
            return "anomaly_agent", None, None

    # Stage 3.5: Consumption and cost queries terminate after Data Agent
    if intent in ("consumption", "cost"):
        return END, None, None

    # Stage 4: Recommendation Agent (for reports or explicit tip requests)
    if intent in ("report", "recommend") and state.get("recommendations") is None:
        return "recommendation_agent", None, None

    # Stage 5: Report Agent
    if intent == "report" and state.get("report") is None:
        return "report_agent", None, None
    if intent == "report":
        return END, None, None

    if intent == "recommend":
        return END, None, None

    # Stage 6: Grounded chat synthesis for forecast/anomaly queries
    messages = state.get("messages", [])
    last_msg = messages[-1] if messages else None
    if last_msg is None or getattr(last_msg, "type", "") != "ai":
        return "chat_agent", None, None

    return END, None, None


def _etl_needs_for_query(query: str) -> tuple[bool, bool]:
    """Determine required ETL pipelines (forecast_etl, anomaly_etl) for the given query."""
    is_anomaly_kw = any(kw in query for kw in ("anomal", "detect"))
    is_report_kw  = any(kw in query for kw in ("report", "summary"))
    if is_report_kw:
        return True, True
    if any(kw in query for kw in ("predict", "forecast")):
        return True, is_anomaly_kw
    if is_anomaly_kw:
        return False, True
    if any(kw in query for kw in ("recommend", "tip", "saving")):
        return True, is_anomaly_kw
    if any(kw in query for kw in _COST_KEYWORDS):
        return True, False
    if any(kw in query for kw in _CONSUMPTION_KEYWORDS) or _QUERY_DATE_RE.search(query):
        return True, False
    return False, False


# ======================================================================= #
#  Graph Builder                                                            #
# ======================================================================= #

async def build_graph():
    """Compile LangGraph StateGraph with Redis-backed checkpointing (falls back to memory)."""
    try:
        redis_client = await get_redis_checkpointer()
        checkpointer = AsyncRedisSaver(
            redis_client=redis_client,
            ttl={"default_ttl": 1440, "refresh_on_read": True},  # 24h TTL
        )
        await checkpointer.asetup()
        logger.info("[Orchestrator] Redis-backed checkpointing enabled (24h TTL).")
    except Exception:
        logger.exception(
            "[Orchestrator] Redis checkpointer unavailable — falling back to "
            "in-memory (conversation state will not survive a restart)."
        )
        checkpointer = MemorySaver()

    builder = StateGraph(AgentState)

    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("data_agent", data_agent)
    builder.add_node("forecast_agent", forecast_agent)
    builder.add_node("anomaly_agent", anomaly_agent)
    builder.add_node("chat_agent", chat_agent)
    builder.add_node("recommendation_agent", recommendation_agent)
    builder.add_node("report_agent", report_agent)

    builder.set_entry_point("orchestrator")

    builder.add_conditional_edges(
        "orchestrator",
        route_from_orchestrator,
        {
            "chat_agent": "chat_agent",
            "data_agent": "data_agent",
            "forecast_agent": "forecast_agent",
            "anomaly_agent": "anomaly_agent",
            "recommendation_agent": "recommendation_agent",
            "report_agent": "report_agent",
            END: END,
        },
    )

    for agent_name in (
        "chat_agent", "data_agent", "forecast_agent",
        "anomaly_agent", "recommendation_agent", "report_agent",
    ):
        builder.add_edge(agent_name, "orchestrator")

    return builder.compile(checkpointer=checkpointer)


_graph = None
_graph_lock = asyncio.Lock()


async def get_graph():
    """Return the compiled graph singleton, initializing on first call."""
    global _graph
    if _graph is None:
        async with _graph_lock:
            if _graph is None:
                _graph = await build_graph()
    return _graph
