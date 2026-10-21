"""
LangGraph node — Chat Agent.
Handles conversational interactions and grounded answers using gathered analytical data.
"""

import logging
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage

from services.llm_service import LLMService
from services.db_service import get_data_latest_date
from core.redis_client import cache_get

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an intelligent energy and water management assistant. "
    "You help users understand their energy and water consumption, "
    "answer questions about efficiency, explain analytics results, "
    "and guide users on how to use the system. "
    "Be concise, friendly, and informative.\n\n"
    "DATES: a 'DATE CONTEXT' line at the end of these instructions always "
    "gives today's real date and the last day the account has data for. "
    "ALWAYS answer date questions ('what is today's date', 'is X in the "
    "past', 'how recent is my data') directly and confidently from it — "
    "never say you lack a clock or point elsewhere for the date.\n\n"
    "GROUNDING: if the user's message ends with an 'ACCOUNT DATA' block, "
    "that block is the ONLY source of truth for their consumption numbers. "
    "Answer from it. If a line says 'unavailable', say that figure could "
    "not be produced right now and suggest the relevant dashboard page — "
    "never estimate or invent a value. If the user asks for saving tips "
    "and none are listed, give 3-5 practical, specific tips grounded in "
    "the figures shown (each starting with a verb). If there is NO ACCOUNT "
    "DATA block and the user asks for a specific consumption/bill figure "
    "you don't have, say so in one line and suggest the Analytics page "
    "(but still answer any date part of the question from DATE CONTEXT). "
    "Otherwise answer conversationally from general knowledge and the "
    "conversation so far.\n\n"
    "BILLING RULES — this system bills Tunisian utilities. All money is in "
    "TND (Tunisian dinar); 1 TND = 1000 millimes. Never quote a flat "
    "per-unit rate, and never use US dollars.\n"
    "Electricity is billed by STEG on PROGRESSIVE MONTHLY brackets (the "
    "meter resets on the 1st of each month). Each bracket's rate applies "
    "only to the portion of the month's cumulative kWh that falls inside "
    "it (hors taxes):\n"
    "  0-50 kWh = 0.062 TND/kWh · 50-100 = 0.096 · 100-200 = 0.176 · "
    "200-300 = 0.218 · 300-500 = 0.341 · above 500 = 0.414 TND/kWh.\n"
    "Water is billed by SONEDE on PROGRESSIVE TRIMESTER brackets (resets "
    "each calendar quarter: Jan-Mar, Apr-Jun, Jul-Sep, Oct-Dec), priced per "
    "1000 litres. Always express water volumes in LITRES (L), never in cubic "
    "metres:\n"
    "  0-20,000 L = 0.200 TND/1000L · 20,000-40,000 = 0.740 · "
    "40,000-70,000 = 1.040 · 70,000-100,000 = 1.490 · "
    "100,000-150,000 = 1.770 · above 150,000 L = 2.310 TND per 1000 L.\n"
    "Because both tariffs are progressive, the cost of any extra usage "
    "depends on how much has ALREADY been consumed in the current billing "
    "period — if you don't have that cumulative figure, say so and give a "
    "range instead of a single number. For an exact bill amount, the app "
    "computes it on the dashboard and when the user asks a cost question "
    "directly."
)

_DOMAIN_LABEL = {"energy": "Electricity", "water": "Water"}
_DOMAIN_UNIT = {"energy": "kWh", "water": "L"}


async def _cached_tips(domains: list) -> list:
    """Retrieve pre-cached recommendations if available for the domain."""
    if len(domains) != 1:
        return []
    try:
        c = await cache_get(f"recommendations:{domains[0]}")
    except Exception:
        return []
    tips = (c or {}).get("recommendations") if isinstance(c, dict) else None
    return tips if isinstance(tips, list) else []


def _facts_from_state(state: dict, cached_tips: list | None = None) -> str:
    """Build the ACCOUNT DATA grounding block from analytical state outputs."""
    domains = [d for d in (state.get("domains") or []) if d in _DOMAIN_LABEL]
    forecasts = state.get("forecasts")
    anomalies = state.get("anomalies")
    summary = state.get("consumption_summary") or {}
    recommendations = state.get("recommendations")
    if recommendations is None and cached_tips:
        recommendations = cached_tips

    if forecasts is None and anomalies is None and not recommendations:
        return ""

    lines: list[str] = []
    for d in domains or list(_DOMAIN_LABEL):
        label, unit = _DOMAIN_LABEL[d], _DOMAIN_UNIT[d]

        if forecasts is not None:
            fv = (forecasts or {}).get(d) or []
            if fv:
                lines.append(
                    f"{label} — next 24 h forecast: about {sum(fv):,.1f} {unit} total, "
                    f"peak hour ~{max(fv):,.2f} {unit}, average ~{sum(fv) / len(fv):,.2f} {unit}/h."
                )
            else:
                lines.append(f"{label} — next 24 h forecast: unavailable.")

        if anomalies is not None:
            av = (anomalies or {}).get(d) or []
            if av:
                head = "; ".join((a.get("description") or "").strip() for a in av[:3] if a.get("description"))
                more = f" (+{len(av) - 3} more)" if len(av) > 3 else ""
                lines.append(f"{label} — {len(av)} recent anomaly event(s): {head}{more}")
            else:
                lines.append(f"{label} — no anomalies flagged recently.")

        s = summary.get(d) or {}
        if s.get("avg_daily"):
            lines.append(
                f"{label} — recent usage: ~{s['avg_daily']:,.1f} {unit}/day "
                f"(last {s.get('record_count', '?')} days, {s.get('total', 0):,.0f} {unit} total)."
            )

    if isinstance(recommendations, list) and recommendations \
            and not (len(recommendations) == 1 and LLMService.is_failure(recommendations[0])):
        lines.append("Saving tips already generated for this account:")
        lines.extend(f"  - {t}" for t in recommendations)

    return "\n".join(lines)


async def chat_agent(state: dict) -> dict:
    """LangGraph node — Chat Agent."""
    messages = state.get("messages", [])

    user_message = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break
    if not user_message:
        logger.warning("[ChatAgent] No HumanMessage found in state — defaulting to Hello.")
        user_message = "Hello"

    # Prior conversation context
    history = []
    for msg in messages[:-1]:
        if isinstance(msg, HumanMessage):
            history.append({"role": "human", "content": msg.content})
        elif isinstance(msg, AIMessage):
            history.append({"role": "ai", "content": msg.content})

    # Temporal context
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d (%A)")
    demo_cap_date = state.get("demo_cap_date")
    try:
        data_through = await get_data_latest_date(cap_date=demo_cap_date)
    except Exception:
        data_through = None
    date_ctx = (
        f"\n\nDATE CONTEXT: Today is {today}. "
        + (f"The account's consumption/anomaly data currently runs through {data_through} — "
           f"anything after that has no readings yet, and 'recent'/'this week' means the days "
           f"up to {data_through}." if data_through else
           "The account has no consumption data loaded.")
    )
    if demo_cap_date:
        date_ctx += (
            f" EXCEPTION: forecast figures (if present in ACCOUNT DATA below) intentionally "
            f"look beyond {demo_cap_date} — that is expected and not a contradiction; only "
            f"historical consumption/anomaly data is capped at {demo_cap_date}."
        )
    system_prompt = SYSTEM_PROMPT + date_ctx

    pipeline_ran = state.get("forecasts") is not None or state.get("anomalies") is not None
    cached_tips = await _cached_tips(state.get("domains") or []) if pipeline_ran else []
    facts = _facts_from_state(state, cached_tips)
    if facts:
        prompt_message = (
            f"{user_message}\n\n"
            f"--- ACCOUNT DATA (answer only from this; say 'unavailable' plainly, never invent) ---\n"
            f"{facts}"
        )
        # Privacy: log only the LENGTH of the user message, never its text.
        logger.info("[ChatAgent] grounded synthesis (history_turns=%d, facts_lines=%d, msg_chars=%d)",
                    len(history), facts.count("\n") + 1, len(user_message))
    else:
        prompt_message = user_message
        logger.info("[ChatAgent] conversation (history_turns=%d, msg_chars=%d)", len(history), len(user_message))

    response_text = await LLMService.chat(
        system_prompt=system_prompt,
        user_message=prompt_message,
        history=history if history else None,
        fast=True,
    )

    # Return AIMessage appended via add_messages reducer
    ai_message = AIMessage(content=response_text)

    agent_log = list(state.get("agent_log", []))
    agent_log.append({
        "agent": "chat_agent",
        "input": user_message,
        "grounded": bool(facts),
        "output": response_text,
    })

    return {
        **state,
        "messages": [ai_message],
        "agent_log": agent_log,
    }
