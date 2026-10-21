"""
STEG (Tunisia) residential electricity progressive monthly tariff service.
"""

STEG_TIERS = [
    (50, 0.062),
    (100, 0.096),
    (200, 0.176),
    (300, 0.218),
    (500, 0.341),
    (float("inf"), 0.414),
]


def compute_tiered_cost(kwh_before: float, kwh_consumed: float) -> float:
    """Calculate tiered STEG cost for consumption given prior cumulative monthly usage."""
    if kwh_consumed <= 0:
        return 0.0
    cursor = kwh_before
    remaining = kwh_consumed
    cost = 0.0
    for up_to, rate in STEG_TIERS:
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


def get_marginal_rate(cumulative_kwh: float) -> float:
    """The rate that applies to the NEXT kWh, given cumulative monthly usage so far."""
    for up_to, rate in STEG_TIERS:
        if cumulative_kwh <= up_to:
            return rate
    return STEG_TIERS[-1][1]


def get_yesterday_cost(records: list) -> dict:
    """Calculate STEG electricity cost for the latest completed day (yesterday)."""
    from datetime import date, timedelta

    day_totals: dict = {}
    for r in records:
        key = r.get("local_date")
        if not key:
            continue
        day_totals[key] = day_totals.get(key, 0.0) + (r.get("total_consumption") or 0.0)

    if not day_totals:
        return {"date": None, "consumption_kwh": 0.0, "cost_dt": 0.0}

    latest_date = max(day_totals.keys())
    yesterday_str = (date.fromisoformat(latest_date) - timedelta(days=1)).isoformat()

    month_prefix = yesterday_str[:7]
    month_before_yesterday = sum(
        v for d, v in day_totals.items() if d.startswith(month_prefix) and d < yesterday_str
    )
    yesterday_total = day_totals.get(yesterday_str, 0.0)
    cost = compute_tiered_cost(month_before_yesterday, yesterday_total)

    return {
        "date": yesterday_str,
        "consumption_kwh": round(yesterday_total, 3),
        "cost_dt": round(cost, 3),
    }


def compute_yesterday_cost_so_far(records: list) -> dict:
    """Calculate yesterday's cost up to the same hour-of-day cutoff as today."""
    from datetime import date, datetime, timedelta

    day_totals: dict = {}
    for r in records:
        key = r.get("local_date")
        if not key:
            continue
        day_totals[key] = day_totals.get(key, 0.0) + (r.get("total_consumption") or 0.0)

    if not day_totals:
        return {"consumption_kwh": 0.0, "cost_dt": 0.0}

    today_str = max(day_totals.keys())
    yesterday_str = (date.fromisoformat(today_str) - timedelta(days=1)).isoformat()
    month_prefix = yesterday_str[:7]

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

    month_before_yesterday = sum(
        v for d, v in day_totals.items() if d.startswith(month_prefix) and d < yesterday_str
    )
    cost = compute_tiered_cost(month_before_yesterday, yesterday_so_far)
    return {"consumption_kwh": round(yesterday_so_far, 3), "cost_dt": round(cost, 3)}
