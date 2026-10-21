from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
import logging
import asyncio
import re
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from core.config import settings
from core.database import init_db
from core.redis_client import close_redis, cache_get, cache_set
from schemas.chat_schema import ChatRequest, ChatResponse

from services.model_service import ModelStore, ANOMALY_DETECTION_DAYS
from services.db_service import get_daily_consumption, get_day_consumption, get_recent_anomalies, get_anomaly_count, get_recent_forecasts, save_forecasts, save_report, get_reports, get_report_by_id, delete_reports, save_anomalies
from services.pdf_service import generate_report_pdf
from services.llm_service import LLMService
from services.forecast_service import compute_forecast, DEMO_PIN_DATE
from agents.orchestrator import get_graph, classify_intent
from agents.data_agent import data_agent
from langchain_core.messages import HumanMessage, AIMessage

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Demo-only date anchors (set to None for live production behavior)
DEMO_CAP_DATE: str | None = None
DEMO_CAP_DATETIME: datetime | None = None


def _report_generated_at() -> datetime | None:
    """Return pinned report generation timestamp when demo mode is active."""
    if DEMO_CAP_DATETIME is not None:
        return DEMO_CAP_DATETIME
    if DEMO_CAP_DATE is None:
        return datetime.strptime(DEMO_PIN_DATE, "%Y-%m-%d")
    return None


def _extract_text(val) -> str:
    """Flatten a LangChain message-content value (str, or list of content blocks) to plain text."""
    if isinstance(val, list):
        return " ".join([b.get("text", "") if isinstance(b, dict) else str(b) for b in val])
    return str(val) if val else ""


# ── Reply formatters for deterministic outputs ────────────────────────────────
def _format_forecast_reply(forecasts: dict, domains: list) -> str:
    parts = []
    for d in domains:
        vals = forecasts.get(d) or []
        if not vals:
            continue
        if d == "energy":
            parts.append(
                f"Energy — next 24h forecast averages {sum(vals)/len(vals):.2f} kWh/h, "
                f"peaking around {max(vals):.2f} kWh/h."
            )
        else:
            parts.append(
                f"Water — next 24h forecast averages {sum(vals)/len(vals):,.0f} L/h, "
                f"peaking around {max(vals):,.0f} L/h."
            )
    return "\n".join(parts) if parts else "No forecast data is available yet."


def _format_anomaly_reply(anomalies: dict, domains: list) -> str:
    parts = []
    for d in domains:
        events = anomalies.get(d) or []
        parts.append(
            f"{d.capitalize()} — {len(events)} anomaly event(s) detected recently."
            if events else f"{d.capitalize()} — no anomalies detected."
        )
    return "\n".join(parts)


_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], start=1)}
_MONTHS.update({m[:3]: i for m, i in list(_MONTHS.items())})


def _parse_query_date(query: str) -> str | None:
    """Parse explicit calendar date from query into 'YYYY-MM-DD' format."""
    q = query.lower()
    now = datetime.now(timezone.utc)
    m = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", q)
    if m:
        y, mo, d = map(int, m.groups())
    else:
        m = re.search(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})\b", q)   # dd/mm/yyyy
        if m:
            d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if y < 100:
                y += 2000
        else:
            m = re.search(r"\b([a-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s*(\d{4}))?", q)
            if m and m.group(1) in _MONTHS:
                mo, d = _MONTHS[m.group(1)], int(m.group(2))
                y = int(m.group(3)) if m.group(3) else now.year
            else:
                m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]{3,9})(?:,?\s*(\d{4}))?", q)
                if not (m and m.group(2) in _MONTHS):
                    return None
                d, mo = int(m.group(1)), _MONTHS[m.group(2)]
                y = int(m.group(3)) if m.group(3) else now.year
    try:
        return datetime(y, mo, d).strftime("%Y-%m-%d")
    except ValueError:
        return None


async def _format_day_consumption_reply(date_str: str, domains: list) -> str:
    """Generate deterministic summary for daily consumption queries."""
    parts = []
    for d in domains:
        site_id = 2 if d == "water" else 1
        day = await get_day_consumption(site_id, date_str)
        unit = "kWh" if d == "energy" else "L"
        label = "Electricity" if d == "energy" else "Water"
        if not day:
            parts.append(f"{label} — no reading recorded for {date_str}.")
        elif d == "energy":
            parts.append(f"{label} — {day['total']:.2f} {unit} on {date_str} "
                         f"(peak hour {day['peak_hour']:.2f} {unit}).")
        else:
            parts.append(f"{label} — {day['total']:,.0f} {unit} on {date_str} "
                         f"(peak hour {day['peak_hour']:,.0f} {unit}).")
    return "\n".join(parts) if parts else f"No reading recorded for {date_str}."


def _format_consumption_reply(consumption_summary: dict, domains: list) -> str:
    parts = []
    for d in domains:
        summary = consumption_summary.get(d) or {}
        if not summary.get("record_count"):
            continue
        n = summary["record_count"]
        if d == "energy":
            parts.append(
                f"Energy — total {summary['total']:.2f} kWh over {n} day(s), "
                f"averaging {summary['avg_daily']:.2f} kWh/day."
            )
        else:
            parts.append(
                f"Water — total {summary['total']:,.0f} L over {n} day(s), "
                f"averaging {summary['avg_daily']:,.0f} L/day."
            )
    return "\n".join(parts) if parts else "No consumption data is available for that period."


_COST_SPAN_UNIT = {"day": 1, "week": 7, "month": 30}


def _parse_cost_span(query: str) -> tuple:
    """Extract time window for tariff cost calculations (past/future days)."""
    import re
    q = (query or "").lower()
    if "yesterday" in q:
        return "past", 1
    if "tomorrow" in q:
        return "future", 1
    if re.search(r"\bthis\s+(month|trimester|quarter|year)\b", q):
        return None, None
    if "this week" in q or "week to date" in q:
        return "past", 7
    mp = re.search(r"\b(?:last|past|previous)\s+(\d+)?\s*(day|week|month)s?\b", q)
    if mp:
        return "past", int(mp.group(1) or 1) * _COST_SPAN_UNIT[mp.group(2)]
    mf = re.search(r"\b(?:next|coming|upcoming|following)\s+(\d+)?\s*(day|week|month)s?\b", q)
    if mf:
        return "future", int(mf.group(1) or 1) * _COST_SPAN_UNIT[mf.group(2)]
    mn = re.search(r"\b(\d+)\s*(day|week|month)s?\b", q)
    if mn:  # bare "5 days" — treat as a forward projection
        return "future", int(mn.group(1)) * _COST_SPAN_UNIT[mn.group(2)]
    return None, None


def _format_cost_reply(consumption_records: dict, domains: list, query: str) -> str:
    """Calculate progressive STEG/SONEDE utility cost for the requested window."""
    from datetime import date, timedelta
    from services import tariff_service, water_tariff_service

    span_kind, span_days = _parse_cost_span(query)

    def _pkey(day_str: str, domain: str) -> str:
        return day_str[:7] if domain == "energy" else water_tariff_service._quarter_prefix(day_str)

    def _window_cost(day_totals: dict, start: str, end: str, domain: str, svc) -> tuple:
        """Compute marginal cost of [start, end] across billing period boundaries."""
        buckets: dict = {}
        for dstr in day_totals:
            if start <= dstr <= end:
                buckets.setdefault(_pkey(dstr, domain), []).append(dstr)
        cost = usage = 0.0
        for pk, dates in buckets.items():
            u = sum(day_totals[x] for x in dates)
            before = sum(v for x, v in day_totals.items()
                         if _pkey(x, domain) == pk and x < min(dates))
            cost += svc.compute_tiered_cost(before, u)
            usage += u
        return cost, usage

    parts = []
    for d in domains:
        records = consumption_records.get(d) or []
        svc = tariff_service if d == "energy" else water_tariff_service
        if d == "energy":
            vol_unit, vfmt, rate_unit, rate_scale = "kWh", ",.1f", "TND/kWh", 1
        else:
            vol_unit, vfmt, rate_unit, rate_scale = "L", ",.0f", "TND per 1000 L", 1000
        period_name = "this month" if d == "energy" else "this trimester"

        day_totals: dict = {}
        for r in records:
            k = r.get("local_date")
            if k:
                day_totals[k] = day_totals.get(k, 0.0) + (r.get("total_consumption") or 0.0)
        if not day_totals:
            continue
        latest = max(day_totals)
        pk = _pkey(latest, d)
        cumulative = sum(v for k, v in day_totals.items() if _pkey(k, d) == pk)
        cost_so_far = svc.compute_tiered_cost(0.0, cumulative)
        marginal = svc.get_marginal_rate(cumulative)

        if span_kind == "past":
            # Calculate cost for completed days
            wend = (date.fromisoformat(latest) - timedelta(days=1)).isoformat()
            wstart = (date.fromisoformat(wend) - timedelta(days=span_days - 1)).isoformat()
            wcost, wusage = _window_cost(day_totals, wstart, wend, d, svc)
            when = "yesterday" if span_days == 1 else f"the last {span_days} days"
            span_txt = f"({wend})" if span_days == 1 else f"({wstart} to {wend})"
            added = "what that day added to" if span_days == 1 else "what those days added to"
            line = (
                f"{d.capitalize()} — {when} {span_txt}: {wusage:{vfmt}} {vol_unit} = "
                f"{wcost:.3f} TND — {added} {period_name.replace('this', 'the current')}'s "
                f"bill at your progressive rate. {period_name.capitalize()} total so far: "
                f"{cumulative:{vfmt}} {vol_unit} = {cost_so_far:.3f} TND."
            )
        else:
            line = (
                f"{d.capitalize()} — {period_name} so far: {cumulative:{vfmt}} {vol_unit} = "
                f"{cost_so_far:.3f} TND. Currently in the {marginal * rate_scale:.3f} {rate_unit} bracket."
            )
            if span_kind == "future":
                completed = sorted(k for k in day_totals if k < latest)[-7:]
                avg_daily = (sum(day_totals[k] for k in completed) / len(completed)
                             if completed else day_totals[latest])
                extra = span_days * avg_daily
                proj = svc.compute_tiered_cost(cumulative, extra)
                line += (
                    f" Estimated next {span_days} day(s): ~{extra:{vfmt}} {vol_unit} ≈ {proj:.3f} TND "
                    f"(your recent daily average, continuing from the current bracket)."
                )
        parts.append(line)

    return "\n".join(parts) if parts else "No consumption data is available to compute a cost."


async def _generate_report(domain: str, *, force: bool = False, source: str = "manual") -> dict:
    """
    Generate or retrieve a utility management report for a specific domain.

    force=True triggers fresh pipeline execution and database persistence.
    """
    cache_key = f"report:{domain}"
    if not force:
        cached = await cache_get(cache_key)
        if cached:
            return cached

    # Coalesce concurrent in-flight generations using Redis lock
    from core.redis_client import get_redis_cache
    lock_key = f"report:inflight:{domain}"
    result_key = f"report:justmade:{domain}"
    holds_lock = False
    if force:
        r = await get_redis_cache()
        holds_lock = bool(await r.set(lock_key, "1", nx=True, ex=240))
        if not holds_lock:
            def _resolve(p):
                if isinstance(p, dict) and p.get("__error__"):
                    raise HTTPException(status_code=int(p["__error__"]),
                                        detail=p.get("detail", "report generation failed"))
                return p
            for _ in range(240):
                await asyncio.sleep(1)
                pending = await cache_get(result_key)
                if pending is not None:
                    return _resolve(pending)
                if not await r.exists(lock_key):
                    break
            pending = await cache_get(result_key)
            if pending is not None:
                return _resolve(pending)
    try:
        graph = await get_graph()
        thread_suffix = f"force_{int(time.time())}" if force else str(int(time.time() // 1800))
        config = {"configurable": {"thread_id": f"report_{domain}_{thread_suffix}"}}
        report_cap_date = DEMO_PIN_DATE if DEMO_CAP_DATE is None else DEMO_CAP_DATE
        final_state = await graph.ainvoke(
            {
                "user_query": f"generate a full {domain} energy management report",
                "messages": [HumanMessage(content=f"generate a full {domain} energy management report")],
                "domains": [domain],
                "demo_cap_date": report_cap_date,
            },
            config=config,
        )

        _NO_CONTENT = "Report generation completed but no content was returned."
        report_text = (
            _extract_text(final_state.get("report"))
            or _extract_text(final_state.get("recommendations"))
            or _NO_CONTENT
        )

        # Verify report generation produced valid content
        if LLMService.is_failure(report_text) or report_text == _NO_CONTENT:
            if force:
                await cache_set(result_key, {"__error__": 502, "detail": _REPORT_UNAVAILABLE}, 30)
            logger.warning("Report for %s produced no usable content: %.200s", domain, report_text)
            raise HTTPException(status_code=502, detail=_REPORT_UNAVAILABLE)

        result = {"report": report_text, "domain": domain}
        await cache_set(cache_key, result, 1800)
        await save_report(domain=domain, report_text=report_text, source=source,
                           generated_at=_report_generated_at())
        if force:
            await cache_set(result_key, result, 30)
        return result
    finally:
        if holds_lock:
            try:
                await (await get_redis_cache()).delete(lock_key)
            except Exception:
                pass


async def _run_weekly_report():
    """Execute weekly report generation and anomaly persistence for all domains."""
    logger.info("[CRON] Triggering weekly automated report generation...")
    for domain in ("energy", "water"):
        try:
            await _generate_report(domain, force=True, source="weekly_cron")
            logger.info("[CRON] Weekly %s report generated and stored in MySQL.", domain)
        except HTTPException as e:
            logger.error("[CRON] Weekly %s report failed (LLM error) — not stored: %s", domain, e.detail)
        except Exception as e:
            logger.error(f"[CRON] Weekly {domain} report failed: {e}")


async def weekly_report_job():
    """Background recurring task that triggers weekly automated report generation."""
    from core.redis_client import cache_get, cache_set
    import time as _time

    WEEK_SECONDS = 7 * 24 * 3600
    LAST_RUN_KEY = "weekly_cron:last_run"

    while True:
        last_run = await cache_get(LAST_RUN_KEY)
        elapsed = (_time.time() - last_run) if last_run else None

        if elapsed is None or elapsed >= WEEK_SECONDS:
            await _run_weekly_report()
            await cache_set(LAST_RUN_KEY, _time.time(), 30 * 24 * 3600)
            sleep_for = WEEK_SECONDS
        else:
            sleep_for = WEEK_SECONDS - elapsed

        try:
            await asyncio.sleep(sleep_for)
        except asyncio.CancelledError:
            break


async def anomaly_timeline_refresh_job():
    """Background recurring task to pre-warm anomaly timeline caches."""
    from core.redis_client import cache_set

    ANOMALY_TIMELINE_TTL = 3600  # 1 hour
    REFRESH_INTERVAL = 55 * 60   # 55 minutes
    while True:
        for domain in ("energy", "water"):
            try:
                timeline = await _compute_anomaly_timeline(domain)
                await cache_set(f"anomaly_timeline:{domain}:{DEMO_CAP_DATE or 'live'}", timeline, ANOMALY_TIMELINE_TTL)
                logger.info("[CRON] Pre-warmed anomaly_timeline:%s (%d points)", domain, len(timeline))
            except Exception as e:
                logger.error(f"[CRON] Anomaly timeline pre-warm failed for {domain}: {e}")
        try:
            await asyncio.sleep(REFRESH_INTERVAL)
        except asyncio.CancelledError:
            break


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup Phase
    logger.info("Initializing MySQL Database...")
    await init_db()
    
    logger.info("Loading ML and DL models into memory (this may take a moment)...")
    ModelStore.load()
    
    logger.info("Pre-compiling LangGraph Orchestrator...")
    await get_graph()
    
    logger.info("Starting weekly cron scheduler...")
    cron_task = asyncio.create_task(weekly_report_job())

    logger.info("Starting anomaly timeline pre-warm scheduler...")
    anomaly_prewarm_task = asyncio.create_task(anomaly_timeline_refresh_job())

    logger.info("Backend is completely ready and listening!")
    yield

    # Shutdown Phase
    logger.info("Cancelling background tasks...")
    cron_task.cancel()
    anomaly_prewarm_task.cancel()
    logger.info("Closing connections...")
    await close_redis()

# Error message constants
_GENERIC_500 = "Internal server error. Please try again later."
_REPORT_UNAVAILABLE = "The report could not be generated right now. Please try again later."

app = FastAPI(
    title="Energy Management Multi-Agent API",
    description="FastAPI gateway for the LangGraph multi-agent system.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def no_browser_cache(request, call_next):
    """
    All /api/* data is dynamic and already has its own server-side Redis
    cache with correct TTLs/invalidation. Without an explicit no-store here,
    browsers are free to apply their own heuristic HTTP caching on top of
    that — silently serving stale JSON indefinitely regardless of what the
    server-side cache says, with no way to force a refresh short of a hard
    reload. This forces every request to actually reach the server.
    """
    response = await call_next(request)
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response

@app.get("/api/health")
async def health_check():
    """Health check for Docker container."""
    return {"status": "healthy", "database": "mysql", "orchestrator": "ready", "models": "loaded"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """
    Main endpoint that accepts a user message and passes it to the LangGraph Orchestrator.
    It uses thread_id to fetch conversation history from Redis checkpointing.
    """
    try:
        graph = await get_graph()
        config = {"configurable": {"thread_id": req.thread_id}}
        
        # Privacy: log only the LENGTH of the message, never its text.
        logger.info(f"Received API request on thread={req.thread_id} ({len(req.message)} chars)")
        
        # Invoke the graph synchronously with per-turn domain derivation.
        final_state = await graph.ainvoke(
            {
                "user_query": req.message,
                "messages": [HumanMessage(content=req.message)],
                "domains": [],
                "demo_cap_date": DEMO_CAP_DATE,
            },
            config=config
        )
        
        def _to_str(val) -> str:
            """Convert LLM response value to a clean string, stripping HTML metadata comments."""
            if isinstance(val, list):
                s = " ".join([b.get("text", "") if isinstance(b, dict) else str(b) for b in val])
            else:
                s = str(val) if val else ""
            return re.sub(r"\s*<!--.*?-->\s*", "", s, flags=re.DOTALL).strip()

        def _plain_text_for_chat(text: str) -> str:
            """Convert markdown formatting to plain prose for chat display."""
            if not text:
                return text
            # Headings -> their own plain line.
            text = re.sub(r"(?m)^#{1,6}\s*(.+)$", r"\1", text)
            # Bold/italic emphasis markers, keeping the wrapped text.
            text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
            text = re.sub(r"(?<!\*)\*(?!\*)([^*\n]+?)\*(?!\*)", r"\1", text)
            # KPI pipe rows ("- label | value | interpretation | status") ->
            # "label — value — interpretation — status".
            def _row(m: "re.Match") -> str:
                cells = [c.strip() for c in m.group(1).split("|") if c.strip()]
                return " — ".join(cells)
            text = re.sub(r"(?m)^-\s*([^\n]*\|[^\n]*)$", _row, text)
            # Remaining "- " bullet markers -> a plain bullet glyph.
            text = re.sub(r"(?m)^-\s+", "• ", text)
            return text.strip()

        # ── Reply selection ─────────────────────────────────────────────
        ai_reply = ""
        reply_already_persisted = False
        intent = classify_intent(final_state.get("user_query") or req.message.lower())
        domains = final_state.get("domains") or ["energy"]

        def _last_ai_text() -> str:
            msgs = final_state.get("messages") or []
            if not msgs:
                return ""
            last = msgs[-1]
            if type(last).__name__ == "AIMessage" or getattr(last, "type", "") == "ai":
                return _to_str(last.content)
            return ""

        # 1. Formal report — the document IS the answer.
        if final_state.get("report"):
            ai_reply = _to_str(final_state["report"])

        # 2. Precise number lookups — deterministic, LLM-free, exact.
        elif intent == "consumption":
            _qdate = _parse_query_date(final_state.get("user_query") or req.message.lower())
            if _qdate:
                ai_reply = await _format_day_consumption_reply(_qdate, domains)
            elif final_state.get("consumption_summary"):
                ai_reply = _format_consumption_reply(final_state["consumption_summary"], domains)
        elif intent == "cost" and final_state.get("consumption_records"):
            ai_reply = _format_cost_reply(
                final_state["consumption_records"], domains,
                final_state.get("user_query") or req.message.lower(),
            )

        # 3. chat_agent's answer.
        else:
            _chat_txt = _last_ai_text()
            if intent == "chat":
                if _chat_txt:
                    ai_reply = _chat_txt
                    reply_already_persisted = True
            elif _chat_txt and not LLMService.is_failure(_chat_txt):
                ai_reply = _chat_txt
                reply_already_persisted = True

        # 4. Data question, LLM unavailable fallback
        if not ai_reply:
            if intent == "forecast" and final_state.get("forecasts") is not None:
                ai_reply = _format_forecast_reply(final_state["forecasts"], domains)
            elif intent == "anomaly" and final_state.get("anomalies") is not None:
                ai_reply = _format_anomaly_reply(final_state["anomalies"], domains)
            elif final_state.get("recommendations"):
                ai_reply = _to_str(final_state["recommendations"])
            if ai_reply and intent in ("forecast", "anomaly") and final_state.get("recommendations"):
                _tips = _to_str(final_state["recommendations"])
                if _tips and not LLMService.is_failure(_tips):
                    ai_reply += "\n\nTips:\n" + _tips

        # Persist assistant reply to conversation history
        if ai_reply and not reply_already_persisted:
            try:
                await graph.aupdate_state(config, {"messages": [AIMessage(content=ai_reply)]})
            except Exception:
                logger.warning("Could not persist assistant reply to thread history", exc_info=True)

        # 5. Fallback
        if not ai_reply:
            ai_reply = "Task completed successfully, but no text response was generated."

        ai_reply = _plain_text_for_chat(ai_reply)

        return ChatResponse(
            thread_id=req.thread_id,
            response=ai_reply
        )
    except Exception as e:
        logger.error(f"Error processing chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_GENERIC_500)

@app.get("/api/dashboard/{domain}")
async def get_dashboard(domain: str, days: int = 30):
    """Retrieve consumption records and summary statistics for dashboard display."""
    cache_key = f"dashboard:{domain}:{days}:{DEMO_CAP_DATE or 'live'}"
    cached = await cache_get(cache_key)
    if cached:
        return cached

    # Pinned date anchor for demo mode
    dash_cap_date = DEMO_PIN_DATE if DEMO_CAP_DATE is None else DEMO_CAP_DATE

    # Dashboard uses forecast-ETL records
    result_state = await data_agent({"domains": [domain], "days": days, "needs_anomaly": False,
                                      "demo_cap_date": dash_cap_date})
    records = result_state["consumption_records"].get(domain, [])
    result = {
        "records": records,
        "summary": result_state["consumption_summary"].get(domain, {}),
    }

    if domain == "energy":
        from services.tariff_service import get_yesterday_cost, compute_yesterday_cost_so_far
        result["yesterday_cost"] = get_yesterday_cost(records)
        result["yesterday_cost_so_far"] = compute_yesterday_cost_so_far(records)

    if domain == "water":
        from services.water_tariff_service import (
            get_yesterday_cost as get_yesterday_water_cost,
            compute_yesterday_cost_so_far as compute_yesterday_water_cost_so_far,
            compute_today_cost_so_far as compute_today_water_cost_so_far,
        )
        # Fetch wider quarterly window for SONEDE water tariff
        tariff_days = max(days, 95)
        if tariff_days == days:
            tariff_records = records
        else:
            tariff_state = await data_agent({"domains": ["water"], "days": tariff_days,
                                              "demo_cap_date": dash_cap_date})
            tariff_records = tariff_state["consumption_records"].get("water", [])
        result["yesterday_cost"] = get_yesterday_water_cost(tariff_records)
        result["yesterday_cost_so_far"] = compute_yesterday_water_cost_so_far(tariff_records)
        result["today_cost_so_far"] = compute_today_water_cost_so_far(tariff_records)

    await cache_set(cache_key, result, 1800)
    return result

@app.get("/api/forecast/{domain}")
async def get_forecast(domain: str):
    """Retrieve 24h-ahead consumption forecast and historical actuals."""
    return await compute_forecast(domain, cap_date=DEMO_CAP_DATE)


async def _compute_anomaly_timeline(domain: str) -> list:
    """Run model inference over calibration window to compute anomaly timeline and events."""
    try:
        cap_date = DEMO_PIN_DATE if DEMO_CAP_DATE is None else DEMO_CAP_DATE
        state = await data_agent(
            {"domains": [domain], "days": ANOMALY_DETECTION_DAYS,
             "needs_forecast": False, "needs_anomaly": True,
             "demo_cap_date": cap_date}
        )
        raw = state["raw_consumption_data"].get(domain, [])
        ts = state["raw_consumption_timestamps"].get(domain, [])

        if raw and len(raw) == len(ts):
            if domain == "water":
                result = ModelStore.detect_water_anomalies(raw, ts)
            else:
                result = ModelStore.detect_energy_anomalies(raw, ts)

            site_id = 2 if domain == "water" else 1
            await save_anomalies(device_id=site_id, events=result["events"], domain=domain)

            # Return the last 30 days (720 hours) for chart rendering
            return result["timeline"][-720:]
        return []
    except Exception as e:
        logger.warning(f"Anomaly timeline computation failed for {domain}: {e}")
        return []


@app.get("/api/anomalies/{domain}")
async def get_anomalies(domain: str):
    """Retrieve detected anomaly events and error timeline for a domain."""
    site_id = 2 if domain == "water" else 1
    anomalies = await get_recent_anomalies(device_id=site_id, domain=domain, limit=200)
    total_count = await get_anomaly_count(device_id=site_id, domain=domain)

    cache_key = f"anomaly_timeline:{domain}:{DEMO_CAP_DATE or 'live'}"
    timeline = await cache_get(cache_key)
    if timeline is None:
        timeline = await _compute_anomaly_timeline(domain)
        await cache_set(cache_key, timeline, 3600)

    return {"anomalies": anomalies, "timeline": timeline, "total_count": total_count}


@app.get("/api/recommendations/{domain}")
async def get_recommendations(domain: str):
    """Generate actionable energy-saving recommendations for a domain."""
    from core.redis_client import cache_get, cache_set

    cache_key = f"recommendations:{domain}"
    cached = await cache_get(cache_key)
    if cached:
        return cached

    try:
        graph = await get_graph()
        config = {"configurable": {"thread_id": f"reco_{domain}_{int(time.time() // 3600)}"}}
        final_state = await graph.ainvoke(
            {
                "user_query": f"give me 5 actionable {domain} energy-saving recommendations based on current consumption data",
                "messages": [HumanMessage(content=f"give me 5 actionable {domain} energy-saving recommendations")],
                "domains": [domain],
                "demo_cap_date": DEMO_CAP_DATE,
            },
            config=config,
        )

        reco_val  = final_state.get("recommendations")
        reco_text = _extract_text(reco_val) or _extract_text(final_state.get("report")) or ""

        if LLMService.is_failure(reco_text):
            logger.warning(f"Recommendations generation failed for {domain}: {reco_text}")
            return {"recommendations": [], "domain": domain}

        tips = reco_val if isinstance(reco_val, list) else ([reco_text] if reco_text else [])
        result = {"recommendations": tips, "domain": domain}
        await cache_set(cache_key, result, 3600)
        return result

    except Exception as e:
        logger.error(f"Recommendations generation failed: {e}", exc_info=True)
        return {"recommendations": [], "domain": domain}


@app.get("/api/weather")
async def get_weather_endpoint():
    """Retrieve observed weather history and forward-looking weather forecast."""
    from core.redis_client import cache_get, cache_set
    import pandas as pd
    from datetime import date as _date, timedelta as _timedelta

    cache_key = "weather:dashboard"
    cached = await cache_get(cache_key)
    if cached:
        return cached

    from services.weather_service import (
        get_observed_weather, get_tomorrow_forecast, get_current_weather,
        seconds_until_midnight_utc,
    )
    observed_df, forecast_df, current_out = await asyncio.gather(
        get_observed_weather(days=8),
        get_tomorrow_forecast(),
        get_current_weather(),
    )

    # Observed 48-hour readings
    observed_out = []
    if not observed_df.empty:
        obs48 = observed_df.tail(48).copy()
        obs48["datetime"] = obs48["datetime"].astype(str)
        observed_out = obs48[["datetime", "temperature", "humidity",
                               "cloud_cover", "solar_radiation"]].to_dict(orient="records")

    # 7-day daily history
    daily_out = []
    if not observed_df.empty:
        odf = observed_df.copy()
        odf["date"] = pd.to_datetime(odf["datetime"]).dt.date
        daily = (odf.groupby("date")
                    .agg(temp_max=("temperature", "max"),
                         temp_min=("temperature", "min"),
                         temp_avg=("temperature", "mean"),
                         humidity=("humidity", "mean"),
                         cloud_cover=("cloud_cover", "mean"))
                    .reset_index())
        today = _date.today()
        daily = daily[daily["date"] <= today].sort_values("date").tail(7)
        for _, row in daily.iterrows():
            daily_out.append({
                "date": row["date"].isoformat(),
                "temp_max": round(float(row["temp_max"]), 1),
                "temp_min": round(float(row["temp_min"]), 1),
                "temp_avg": round(float(row["temp_avg"]), 1),
                "humidity": round(float(row["humidity"]), 1),
                "cloud_cover": round(float(row["cloud_cover"]), 1),
            })

    # Tomorrow's forecast
    tomorrow_out = None
    if not forecast_df.empty:
        fdf = forecast_df.copy()
        fdf["date"] = pd.to_datetime(fdf["datetime"]).dt.date
        tomorrow_date = _date.today() + _timedelta(days=1)
        tmw = fdf[fdf["date"] == tomorrow_date]
        if not tmw.empty:
            tomorrow_out = {
                "date": tomorrow_date.isoformat(),
                "temp_max": round(float(tmw["temperature"].max()), 1),
                "temp_min": round(float(tmw["temperature"].min()), 1),
                "temp_avg": round(float(tmw["temperature"].mean()), 1),
                "humidity": round(float(tmw["humidity"].mean()), 1),
                "cloud_cover": round(float(tmw["cloud_cover"].mean()), 1),
            }

    result = {"observed": observed_out, "daily": daily_out, "tomorrow": tomorrow_out, "current": current_out}
    await cache_set(cache_key, result, min(3600, seconds_until_midnight_utc()))
    return result


@app.get("/api/report/{domain}")
async def get_report(domain: str):
    """Generate fresh report on demand and persist to database."""
    try:
        return await _generate_report(domain, force=True)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Report generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_GENERIC_500)


@app.get("/api/report/{domain}/pdf")
async def get_report_pdf(domain: str):
    """Generate report and return as downloadable PDF document."""
    try:
        result = await _generate_report(domain)
        pdf_bytes = generate_report_pdf(result["report"], domain=domain, source="manual",
                                         generated_at=_report_generated_at())
        filename = f"sfmconnect-{domain}-report-{int(time.time())}.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as e:
        logger.error(f"Report PDF generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_GENERIC_500)


@app.get("/api/reports/{domain}")
async def list_reports(domain: str):
    """Retrieve history of past reports for a domain."""
    return {"reports": await get_reports(domain)}


@app.delete("/api/reports")
async def delete_reports_endpoint(ids: str):
    """Bulk-delete stored reports by comma-separated ID list."""
    try:
        report_ids = [int(x) for x in ids.split(",") if x.strip()]
    except ValueError:
        raise HTTPException(status_code=400, detail="ids must be a comma-separated list of integers")
    deleted = await delete_reports(report_ids)
    return {"deleted": deleted}


@app.get("/api/reports/{report_id}/pdf")
async def get_stored_report_pdf(report_id: int):
    """Render a previously stored report by ID as a downloadable PDF."""
    report = await get_report_by_id(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail=f"Report {report_id} not found")

    try:
        pdf_bytes = generate_report_pdf(report.report_text, domain=report.domain,
                                        generated_at=report.generated_at, source=report.source)
        filename = f"sfmconnect-{report.domain}-report-{report_id}.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as e:
        logger.error(f"Stored report PDF render failed for id={report_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=_GENERIC_500)

