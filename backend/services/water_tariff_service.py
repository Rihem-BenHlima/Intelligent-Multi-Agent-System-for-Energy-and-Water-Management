"""
SONEDE (Tunisia) residential water progressive quarterly tariff service.
All measurements are calculated in litres.
"""

# (upper bound in LITRES, rate in TND per LITRE)
WATER_TIERS = [
    (20_000,       0.000200),
    (40_000,       0.000740),
    (70_000,       0.001040),
    (100_000,      0.001490),
    (150_000,      0.001770),
    (float("inf"), 0.002310),
]


def compute_tiered_cost(l_before: float, l_consumed: float) -> float:
    """Calculate tiered SONEDE cost for consumption given prior cumulative quarterly usage."""
    if l_consumed <= 0:
        return 0.0
    cursor = l_before
    remaining = l_consumed
    cost = 0.0
    for up_to, rate in WATER_TIERS:
        if remaining <= 0:
            break
        if cursor >= up_to:
            continue
        room_in_tier = up_to - cursor
        amount = min(remaining, room_in_tier)
        cost += amount * rate
        cursor += amount
        remaining -= amount
    return cost


def get_marginal_rate(cumulative_l: float) -> float:
    """The TND-per-litre rate that applies to the NEXT litre, given cumulative trimester usage."""
    for up_to, rate in WATER_TIERS:
        if cumulative_l <= up_to:
            return rate
    return WATER_TIERS[-1][1]


def _quarter_prefix(date_str: str) -> str:
    """Return calendar quarter prefix for a date string (e.g., '2026-07-22' -> '2026-Q3')."""
    year = date_str[:4]
    month = int(date_str[5:7])
    quarter = (month - 1) // 3 + 1
    return f"{year}-Q{quarter}"


def get_yesterday_cost(records: list) -> dict:
    """Calculate SONEDE water cost for the latest completed day (yesterday)."""
    from datetime import date, timedelta

    day_totals: dict = {}
    for r in records:
        key = r.get("local_date")
        if not key:
            continue
        day_totals[key] = day_totals.get(key, 0.0) + (r.get("total_consumption") or 0.0)

    if not day_totals:
        return {"date": None, "consumption_l": 0.0, "cost_dt": 0.0}

    latest_date = max(day_totals.keys())
    yesterday_str = (date.fromisoformat(latest_date) - timedelta(days=1)).isoformat()

    quarter_key = _quarter_prefix(yesterday_str)
    quarter_before_yesterday = sum(
        v for d, v in day_totals.items()
        if _quarter_prefix(d) == quarter_key and d < yesterday_str
    )
    yesterday_total = day_totals.get(yesterday_str, 0.0)
    cost = compute_tiered_cost(quarter_before_yesterday, yesterday_total)

    return {
        "date": yesterday_str,
        "consumption_l": round(yesterday_total, 3),
        "cost_dt": round(cost, 3),
    }


def compute_yesterday_cost_so_far(records: list) -> dict:
    """Calculate yesterday's water cost up to the same hour-of-day cutoff as today."""
    from datetime import date, datetime, timedelta

    day_totals: dict = {}
    for r in records:
        key = r.get("local_date")
        if not key:
            continue
        day_totals[key] = day_totals.get(key, 0.0) + (r.get("total_consumption") or 0.0)

    if not day_totals:
        return {"consumption_l": 0.0, "cost_dt": 0.0}

    today_str = max(day_totals.keys())
    yesterday_str = (date.fromisoformat(today_str) - timedelta(days=1)).isoformat()
    quarter_key = _quarter_prefix(yesterday_str)

    today_hours: list = []
    for r in records:
        if r.get("local_date") == today_str:
            today_hours.append(datetime.utcfromtimestamp(r["date"] / 1000).hour)
    cutoff_hour = max(today_hours) if today_hours else 23

    yesterday_so_far = 0.0
    for r in records:
        key = r.get("local_date")
        if key == yesterday_str:
            v = r.get("total_consumption") or 0.0
            ts = datetime.utcfromtimestamp(r["date"] / 1000)
            if ts.hour <= cutoff_hour:
                yesterday_so_far += v

    quarter_before_yesterday = sum(
        v for d, v in day_totals.items()
        if _quarter_prefix(d) == quarter_key and d < yesterday_str
    )
    cost = compute_tiered_cost(quarter_before_yesterday, yesterday_so_far)
    return {"consumption_l": round(yesterday_so_far, 3), "cost_dt": round(cost, 3)}


def compute_today_cost_so_far(records: list) -> dict:
    """Calculate today's SONEDE water cost so far against cumulative quarterly usage."""
    day_totals: dict = {}
    for r in records:
        key = r.get("local_date")
        if not key:
            continue
        day_totals[key] = day_totals.get(key, 0.0) + (r.get("total_consumption") or 0.0)

    if not day_totals:
        return {"date": None, "consumption_l": 0.0, "cost_dt": 0.0}

    today_str = max(day_totals.keys())
    quarter_key = _quarter_prefix(today_str)

    quarter_before_today = sum(
        v for d, v in day_totals.items()
        if _quarter_prefix(d) == quarter_key and d < today_str
    )
    today_so_far = day_totals.get(today_str, 0.0)
    cost = compute_tiered_cost(quarter_before_today, today_so_far)

    return {
        "date": today_str,
        "consumption_l": round(today_so_far, 3),
        "cost_dt": round(cost, 3),
    }
