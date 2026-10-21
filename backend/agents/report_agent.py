"""
LangGraph node — Report Agent.
Compiles deterministic summary metrics, anomaly events, forecasts,
and recommendations into a formatted periodic consumption report.
"""

import json
import logging
from datetime import datetime, timezone

from services.llm_service import LLMService
from core.redis_client import cache_get, cache_set

logger = logging.getLogger(__name__)

_REPORT_CACHE_TTL = 1800  # 30 min — matches main._generate_report

_REPORT_WINDOW_DAYS = 7        # "this week" — the headline period
_ANOMALY_LOOKBACK_DAYS = 30    # anomalies newer than this = "recent", shown in full
_DAY_MS = 86_400_000


def _flatten_text(val) -> str:
    """Flatten message content to string."""
    if isinstance(val, list):
        return " ".join([b.get("text", "") if isinstance(b, dict) else str(b) for b in val])
    return str(val) if val else ""


def _n(count: int, singular: str, plural: str | None = None) -> str:
    """Format count with singular or plural noun."""
    return f"{count} {singular if count == 1 else (plural or singular + 's')}"


def _iso_to_ms(s):
    try:
        return datetime.fromisoformat(str(s).replace("Z", "")).timestamp() * 1000
    except (TypeError, ValueError):
        return None


def _fmt_date(iso) -> str:
    ms = _iso_to_ms(iso)
    if ms is None:
        return str(iso)
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%d %b %Y")


def _domain_facts(domain: str, consumption_data: dict, consumption_summary: dict,
                  consumption_records: dict, forecasts: dict, anomalies: dict) -> dict:
    """Pre-compute summary metrics and chart series for a domain."""
    unit = "kWh" if domain == "energy" else "L"
    data = consumption_data.get(domain) or []
    summ = consumption_summary.get(domain) or {}
    records = consumption_records.get(domain) or []
    f_vals = [v for v in (forecasts.get(domain) or []) if isinstance(v, (int, float))]
    a_all = [ev for ev in (anomalies.get(domain) or []) if isinstance(ev, dict)]

    anchor_ms = max((r.get("date") or 0) for r in records) if records else None

    week = data[-24 * _REPORT_WINDOW_DAYS:] if data else []
    day = data[-24:] if data else []
    week_total = float(sum(week))
    day_total = float(sum(day))
    day_peak = float(max(day)) if day else 0.0
    week_avg_daily = week_total / _REPORT_WINDOW_DAYS if week else 0.0

    daily_7 = []
    if len(week) >= 24:
        n_days = len(week) // 24
        last_dow = (datetime.fromtimestamp(anchor_ms / 1000, tz=timezone.utc).weekday()
                    if anchor_ms else 0)
        names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        for i in range(n_days):
            chunk = week[i * 24:(i + 1) * 24]
            dow = (last_dow - (n_days - 1 - i)) % 7
            daily_7.append([names[dow], round(float(sum(chunk)), 1)])
    # tod_week: usage by time of day across the reporting week.
    tod_buckets = {"Night": 0.0, "Morning": 0.0, "Afternoon": 0.0, "Evening": 0.0}
    for idx, v in enumerate(week):
        h = (len(week) - 1 - idx) % 24  # hour-of-day, counting back from newest
        if h < 6:      tod_buckets["Night"] += float(v)
        elif h < 12:   tod_buckets["Morning"] += float(v)
        elif h < 18:   tod_buckets["Afternoon"] += float(v)
        else:          tod_buckets["Evening"] += float(v)
    tod_week = [[k, round(x, 1)] for k, x in tod_buckets.items()]

    avg_daily = float(summ.get("avg_daily") or 0.0)
    max_daily = float(summ.get("max_daily") or 0.0)
    hist_days = int(summ.get("record_count") or 0)
    vs_norm_pct = ((week_avg_daily - avg_daily) / avg_daily * 100) if avg_daily > 0 else None

    f_total = float(sum(f_vals)) if f_vals else 0.0
    f_peak = float(max(f_vals)) if f_vals else 0.0

    recent, older = [], 0
    cutoff = (anchor_ms - _ANOMALY_LOOKBACK_DAYS * _DAY_MS) if anchor_ms is not None else None
    for ev in a_all:
        end_ms = _iso_to_ms(ev.get("end"))
        if cutoff is not None and end_ms is not None and end_ms >= cutoff:
            d = (ev.get("description") or "").lower()
            direction = "spike" if "higher" in d else "drop" if "lower" in d else "unclear"
            recent.append({**ev, "direction": direction})
        else:
            older += 1

    return {
        "domain": domain, "unit": unit,
        "week_total": week_total, "week_avg_daily": week_avg_daily,
        "day_total": day_total, "day_peak": day_peak,
        "avg_daily": avg_daily, "max_daily": max_daily, "hist_days": hist_days,
        "vs_norm_pct": vs_norm_pct,
        "f_total": f_total, "f_peak": f_peak, "has_forecast": bool(f_vals),
        "recent_anomalies": recent, "older_anomalies": older,
        "spikes": [e for e in recent if e["direction"] == "spike"],
        "drops": [e for e in recent if e["direction"] == "drop"],
        "has_data": bool(data),
        "daily_7": daily_7, "tod_week": tod_week,
    }


def _verdict(f: dict) -> str:
    if f["spikes"]:
        return "Needs attention"
    p = f["vs_norm_pct"]
    if p is not None and p > 60:
        return "Higher than usual"
    if p is not None and p < -60:
        return "Lower than usual"
    if f["drops"]:
        return "Normal, with quiet spells"
    return "Normal"


def _summary_block(f: dict) -> str:
    """Build deterministic summary section with KPI cards."""
    unit = f["unit"]
    if not f["has_data"]:
        return "## Summary\n**No data.** No consumption data was available for this period."

    verdict = _verdict(f)
    p = f["vs_norm_pct"]
    if p is None:
        cmp = "There is no historical average yet to compare against"
    elif abs(p) <= 15:
        cmp = f"That is in line with the usual {f['avg_daily']:,.0f} {unit} per day"
    else:
        cmp = (f"That is about {abs(p):.0f}% {'above' if p > 0 else 'below'} "
               f"the usual {f['avg_daily']:,.0f} {unit} per day")

    n_spk, n_drp = len(f["spikes"]), len(f["drops"])
    if n_spk:
        an = (f" {_n(n_spk, 'usage spike')} in the last {_ANOMALY_LOOKBACK_DAYS} days "
              f"{'needs' if n_spk == 1 else 'need'} review.")
    elif n_drp:
        an = (f" {_n(n_drp, 'period')} of unusually low usage {'was' if n_drp == 1 else 'were'} "
              f"flagged — actual consumption fell below the level the detector expected.")
    else:
        an = f" No anomalies were flagged in the last {_ANOMALY_LOOKBACK_DAYS} days."

    verdict_line = (
        f"**{verdict}.** Over the last {_REPORT_WINDOW_DAYS} days the building used "
        f"{f['week_total']:,.0f} {unit}. {cmp}.{an}"
    )

    week_norm = f["avg_daily"] * _REPORT_WINDOW_DAYS
    if p is None:
        wk_interp, wk_status = "no baseline to compare against", "normal"
    else:
        wk_interp = (f"{abs(p):.0f}% {'above' if p > 0 else 'below'} the usual "
                     f"~{week_norm:,.0f} {unit}/week")
        wk_status = "alert" if f["spikes"] else "watch" if abs(p) > 60 else "normal"
    wk = f"- This week | {f['week_total']:,.0f} {unit} | {wk_interp} | {wk_status}"

    if f["avg_daily"] > 0:
        ratio = f["day_total"] / f["avg_daily"]
        if f["day_total"] >= f["max_daily"] * 0.9:
            d_interp, d_status = f"near the busiest day on record ({f['max_daily']:,.0f} {unit})", "watch"
        elif ratio < 0.3:
            d_interp, d_status = f"well below a normal day (~{f['avg_daily']:,.0f} {unit})", "watch"
        else:
            d_interp, d_status = f"about {ratio * 100:.0f}% of a normal day", "normal"
    else:
        d_interp, d_status = "no daily baseline", "normal"
    d = f"- Last 24 hours | {f['day_total']:,.0f} {unit} | {d_interp} | {d_status}"

    if f["has_forecast"]:
        if f["day_total"] > 0 and f["f_total"] > f["day_total"] * 1.5:
            fc_interp = "well above the last 24 h — normal activity expected to resume"
        elif f["day_total"] > 0 and f["f_total"] < f["day_total"] * 0.66:
            fc_interp = "below the last 24 h — the quiet spell is expected to continue"
        else:
            fc_interp = "in line with recent usage"
        fc = f"- Next 24 hours (forecast) | {f['f_total']:,.0f} {unit} | {fc_interp} | normal"
    else:
        fc = "- Next 24 hours (forecast) | n/a | forecast unavailable this run | watch"

    if n_spk:
        an_interp, an_status = f"{_n(n_spk, 'spike')} above the expected level — worth checking", "alert"
    elif n_drp:
        an_interp, an_status = f"{_n(n_drp, 'period')} below the detector's expected level", "watch"
    else:
        an_interp, an_status = "none flagged in the monitoring window", "normal"
    if f["older_anomalies"]:
        an_interp += f"; {f['older_anomalies']} older on record"
    an = (f"- Anomalies (last {_ANOMALY_LOOKBACK_DAYS} days) | {len(f['recent_anomalies'])} "
          f"| {an_interp} | {an_status}")

    return "## Summary\n" + verdict_line + "\n\n" + "\n".join([wk, d, fc, an])


def _facts_for_prompt(f: dict) -> str:
    unit = f["unit"]
    lines = [
        f"DOMAIN: {f['domain']} (all volumes in {unit})",
        f"Verdict already stated in the Summary (stay consistent): {_verdict(f)}",
        f"Last 24 h total: {f['day_total']:,.1f} {unit}",
        (f"Next 24 h forecast total: {f['f_total']:,.1f} {unit}, busiest hour {f['f_peak']:,.2f} {unit}"
         if f["has_forecast"] else "Next 24 h forecast: unavailable"),
    ]
    if f["recent_anomalies"]:
        lines.append(f"Recent anomalies (last {_ANOMALY_LOOKBACK_DAYS} days), {len(f['recent_anomalies'])} — "
                     "each description already contains the detector's expected-vs-actual figures:")
        for ev in f["recent_anomalies"]:
            lines.append(f"  - {_fmt_date(ev.get('start'))} [{ev['direction']}]: "
                         f"{(ev.get('description') or '').strip()}")
    else:
        lines.append(f"Recent anomalies (last {_ANOMALY_LOOKBACK_DAYS} days): none")
    lines.append(f"Older anomalies on record (before this period): {f['older_anomalies']}")
    return "\n".join(lines)


def _fallback_anomalies_md(f: dict) -> str:
    """Fallback anomalies markdown on LLM failure."""
    lines = ["## Anomalies"]
    if f["recent_anomalies"]:
        for ev in f["recent_anomalies"]:
            desc = (ev.get("description") or "").strip().rstrip(".")
            lines.append(f"- {_fmt_date(ev.get('start'))} — {desc}.")
    else:
        lines.append(f"- No anomalies in the last {_ANOMALY_LOOKBACK_DAYS} days.")
    if f["older_anomalies"]:
        lines.append(f"- {_n(f['older_anomalies'], 'earlier event')} on record from before this period.")
    return "\n".join(lines)


def _fallback_forecast_md(f: dict) -> str:
    """Fallback forecast markdown on LLM failure."""
    unit = f["unit"]
    if not f["has_forecast"]:
        return "## Forecast\n- A forecast is not available for this period."
    if f["day_total"] > 0 and f["f_total"] > f["day_total"] * 1.5:
        cmp = "well above the last 24 hours — normal activity is expected to resume"
    elif f["day_total"] > 0 and f["f_total"] < f["day_total"] * 0.66:
        cmp = "below the last 24 hours — the quiet spell is expected to continue"
    else:
        cmp = "in line with recent usage"
    return (
        "## Forecast\n"
        f"- Next 24 hours: about {f['f_total']:,.0f} {unit} in total, busiest hour near "
        f"{f['f_peak']:,.1f} {unit}.\n"
        f"- That is {cmp} (last 24 h: {f['day_total']:,.0f} {unit})."
    )


SYSTEM_PROMPT = (
    "You are a utilities analyst writing for a building manager who is NOT "
    "technical. Explain plainly; do not just restate figures.\n\n"
    "Use ONLY the numbers provided — never recompute or invent figures, and "
    "never invent an anomaly severity rating.\n\n"
    "OUTPUT: Markdown. Exactly these two level-2 headings, in this order, "
    "written verbatim:\n"
    "## Anomalies\n"
    "## Forecast\n"
    "Nothing else — a Summary and a Dashboard are already prepended, and "
    "Recommendations are appended after you. Do NOT add any other heading.\n\n"
    "## Anomalies — the anomaly detector reconstructs what it EXPECTED for "
    "each hour and compares that to what actually happened; those expected-"
    "vs-actual figures are already inside each item's description. For every "
    "recent anomaly, write one bullet: the date, how long it lasted, and — "
    "quoting the description's own figures — whether actual usage was ABOVE "
    "what the detector expected (possible leak or fault, worth checking) or "
    "BELOW it. Do NOT speculate about the cause beyond that. "
    "If there are none: a single bullet '- No anomalies in the last 30 days.' "
    "Then, only if older events remain, add: '- N earlier events remain on record.'\n\n"
    "## Forecast — 1 to 2 short sentences: the next-24h total and how it "
    "compares to the last 24 hours. If it is much higher, say normal "
    "activity is expected to resume; if much lower, the quiet period is "
    "expected to continue.\n\n"
    "RULES: Water is always LITRES (L), never cubic metres. Electricity is "
    "kWh. Be concise — no padding, no meta-commentary."
)


async def report_agent(state: dict) -> dict:
    """LangGraph node — Report Agent."""
    domains             = state.get("domains") or ["energy"]

    # Check cache for single-domain reports
    if len(domains) == 1:
        try:
            cached = await cache_get(f"report:{domains[0]}")
        except Exception:
            cached = None
        cached_text = (cached or {}).get("report") if isinstance(cached, dict) else None
        if cached_text and not LLMService.is_failure(cached_text):
            logger.info("[ReportAgent] reusing cached report for domain=%s", domains[0])
            agent_log = list(state.get("agent_log", []))
            agent_log.append({"agent": "report_agent",
                              "input": {"domains": domains, "source": "cache"},
                              "output": cached_text})
            return {**state, "report": cached_text, "agent_log": agent_log}

    consumption_data    = state.get("consumption_data") or {}
    consumption_summary = state.get("consumption_summary") or {}
    consumption_records = state.get("consumption_records") or {}
    forecasts           = state.get("forecasts") or {}
    anomalies           = state.get("anomalies") or {}
    recommendations     = state.get("recommendations")

    facts = [
        _domain_facts(d, consumption_data, consumption_summary, consumption_records,
                      forecasts, anomalies)
        for d in domains
    ]

    summary_md = "\n\n".join(_summary_block(f) for f in facts)
    prompt_facts = "\n\n".join(_facts_for_prompt(f) for f in facts)

    # Chart series metadata comment for PDF rendering
    chart_payload = {
        "domains": {
            f["domain"]: {"unit": f["unit"], "daily_7": f["daily_7"], "tod_week": f["tod_week"]}
            for f in facts
        }
    }
    chart_comment = f"\n\n<!--SFMDATA {json.dumps(chart_payload, separators=(',', ':'))}-->"

    user_message = (
        f"Write the report body for: {', '.join(domains)}.\n\n{prompt_facts}"
    )

    logger.info("[ReportAgent] Generating report for domains=%s", domains)

    analysis_flat = _flatten_text(await LLMService.chat(
        system_prompt=SYSTEM_PROMPT, user_message=user_message)).strip()

    # Recommendations section
    if isinstance(recommendations, list):
        rec_lines = [t.strip() for t in recommendations if isinstance(t, str) and t.strip()]
    else:
        flat = _flatten_text(recommendations).strip()
        rec_lines = [flat] if flat else []

    if len(rec_lines) == 1 and LLMService.is_failure(rec_lines[0]):
        rec_block = "_Recommendations could not be generated this run._"
    elif rec_lines:
        rec_block = "\n".join(f"- {t}" for t in rec_lines)
    else:
        rec_block = "_No recommendations available._"

    llm_failed = LLMService.is_failure(analysis_flat)
    if llm_failed:
        # Fallback deterministic report body on LLM failure
        logger.warning("[ReportAgent] LLM unavailable — emitting report with deterministic "
                       "Anomalies/Forecast sections")
        body = "\n\n".join(
            _fallback_anomalies_md(f) + "\n\n" + _fallback_forecast_md(f) for f in facts
        )
        note = ("\n\n_Note: the written analysis service was unavailable when this report "
                "was generated. All figures above are current; the sections below are "
                "auto-generated from the same data._")
        report_text = (
            f"{summary_md}{note}\n\n"
            f"{body}\n\n"
            f"## Recommendations\n{rec_block}"
            f"{chart_comment}"
        )
    else:
        report_text = (
            f"{summary_md}\n\n"
            f"{analysis_flat}\n\n"
            f"## Recommendations\n{rec_block}"
            f"{chart_comment}"
        )

    # Cache report for single-domain queries
    if len(domains) == 1 and not LLMService.is_failure(report_text):
        try:
            await cache_set(f"report:{domains[0]}",
                            {"report": report_text, "domain": domains[0]},
                            _REPORT_CACHE_TTL)
        except Exception as exc:
            logger.warning("[ReportAgent] report cache_set failed for %s: %s", domains[0], exc)

    agent_log = list(state.get("agent_log", []))
    agent_log.append({
        "agent": "report_agent",
        "input": {
            "domains":         domains,
            "anomalies_shown": {f["domain"]: len(f["recent_anomalies"]) for f in facts},
            "anomalies_older": {f["domain"]: f["older_anomalies"] for f in facts},
            "verdict":         {f["domain"]: _verdict(f) for f in facts},
            "llm_failed":      llm_failed,
        },
        "output": report_text,
    })

    return {**state, "report": report_text, "agent_log": agent_log}
