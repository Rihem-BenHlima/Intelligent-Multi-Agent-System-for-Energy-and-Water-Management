import json
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc, and_, delete

from core.database import AsyncSessionLocal
from core.redis_client import cache_get, cache_set, cache_delete
from models.db_models import (
    Client, Site, Device, Channel, ChannelVoltix,
    Consumption, DailyConsumption, Anomaly, Alarm, Forecast, Report
)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def get_db_session():
    """Provides a transactional scope around a series of operations."""
    session = AsyncSessionLocal()
    try:
        yield session
    finally:
        await session.close()

async def get_daily_consumption(site_id: int, days: int = 30, cap_date: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve daily consumption data for a site over the given window."""
    cache_key = f"consumption:{site_id}:{days}:{cap_date or 'live'}"
    cached_data = await cache_get(cache_key)
    if cached_data:
        return cached_data

    async with get_db_session() as session:
        try:
            # Anchor window to latest available record date for the site
            latest_stmt = select(func.max(DailyConsumption.date)).where(DailyConsumption.site_id == site_id)
            if cap_date:
                latest_stmt = latest_stmt.where(DailyConsumption.local_date <= cap_date)
            latest_ms = await session.scalar(latest_stmt)
            if latest_ms is None:
                await cache_set(cache_key, [], ttl=1800)
                return []
            cutoff_ms = latest_ms - int(days * 24 * 3600 * 1000)

            stmt = (
                select(DailyConsumption)
                .where(
                    and_(
                        DailyConsumption.site_id == site_id,
                        DailyConsumption.date >= cutoff_ms,
                        DailyConsumption.date <= latest_ms,
                    )
                )
                .order_by(desc(DailyConsumption.date))
            )
            result = await session.execute(stmt)
            records = result.scalars().all()
            
            data = [
                {
                    "date": r.date,
                    "local_date": str(r.local_date) if r.local_date else None,
                    "total_consumption": r.total_consumption
                }
                for r in records
            ]
            
            # Cache for 30 minutes
            await cache_set(cache_key, data, ttl=1800)
            return data
        except Exception as e:
            logger.error(f"Error fetching daily consumption for site {site_id}: {e}")
            return []

async def get_recent_anomalies(device_id: Optional[int] = None, domain: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
    """Retrieve recent anomalies, optionally filtered by device_id and domain."""
    cache_key = f"anomalies:{device_id}:{domain}:{limit}"
    cached_data = await cache_get(cache_key)
    if cached_data:
        return cached_data

    async with get_db_session() as session:
        try:
            stmt = select(Anomaly).order_by(desc(Anomaly.start_date)).limit(limit)
            if device_id is not None:
                stmt = stmt.where(Anomaly.device_id == device_id)
            if domain is not None:
                stmt = stmt.where(Anomaly.project == domain)
                
            result = await session.execute(stmt)
            records = result.scalars().all()
            
            data = [
                {
                    "id": r.id,
                    "anomaly_name": r.anomaly_name,
                    "start_date": r.start_date,
                    "end_date": r.end_date,
                    "description": r.description,
                }
                for r in records
            ]
            
            # Cache for 30 mins
            await cache_set(cache_key, data, ttl=1800)
            return data
        except Exception as e:
            logger.error(f"Error fetching anomalies for device {device_id}: {e}")
            return []

async def get_anomaly_count(device_id: Optional[int] = None, domain: Optional[str] = None) -> int:
    """Return total count of anomaly records for a device and domain."""
    cache_key = f"anomaly_count:{device_id}:{domain}"
    cached = await cache_get(cache_key)
    if cached is not None:
        return cached

    async with get_db_session() as session:
        try:
            stmt = select(func.count()).select_from(Anomaly)
            if device_id is not None:
                stmt = stmt.where(Anomaly.device_id == device_id)
            if domain is not None:
                stmt = stmt.where(Anomaly.project == domain)
            result = await session.execute(stmt)
            count = result.scalar() or 0
            await cache_set(cache_key, count, ttl=1800)
            return count
        except Exception as e:
            logger.error(f"Error counting anomalies for device {device_id}: {e}")
            return 0

async def get_clients() -> List[Dict[str, Any]]:
    """Retrieve all clients."""
    cache_key = "clients:all"
    cached_data = await cache_get(cache_key)
    if cached_data:
        return cached_data

    async with get_db_session() as session:
        try:
            stmt = select(Client)
            result = await session.execute(stmt)
            records = result.scalars().all()
            
            data = [{"id": r.id, "name": r.name} for r in records]
            
            # Cache for 6 hours
            await cache_set(cache_key, data, ttl=21600)
            return data
        except Exception as e:
            logger.error(f"Error fetching clients: {e}")
            return []

async def get_devices_for_site(site_id: int) -> List[Dict[str, Any]]:
    """Retrieve devices for a specific site."""
    async with get_db_session() as session:
        try:
            stmt = select(Device).where(Device.site_id == site_id)
            result = await session.execute(stmt)
            records = result.scalars().all()
            
            data = [
                {
                    "id": r.id,
                    "serial_number": r.serial_number,
                    "enable": r.enable,
                    "last_connection": str(r.last_connection) if r.last_connection else None
                }
                for r in records
            ]
            return data
        except Exception as e:
            logger.error(f"Error fetching devices for site {site_id}: {e}")
            return []

async def get_data_latest_date(cap_date: Optional[str] = None) -> Optional[str]:
    """Return the most recent consumption date ('YYYY-MM-DD')."""
    if cap_date:
        return cap_date
    cached = await cache_get("data:latest_date")
    if cached:
        return cached
    async with get_db_session() as session:
        try:
            d = await session.scalar(select(func.max(DailyConsumption.local_date)))
            if d is None:
                return None
            s = d.isoformat() if hasattr(d, "isoformat") else str(d)
            await cache_set("data:latest_date", s, ttl=3600)
            return s
        except Exception as e:
            logger.error(f"get_data_latest_date: {e}")
            return None


async def get_day_consumption(site_id: int, local_date: str) -> Optional[Dict[str, Any]]:
    """Return total consumption metrics for a specific calendar day ('YYYY-MM-DD')."""
    async with get_db_session() as session:
        try:
            row = (await session.execute(
                select(
                    func.sum(DailyConsumption.total_consumption).label("total"),
                    func.count(DailyConsumption.id).label("hours"),
                    func.max(DailyConsumption.total_consumption).label("peak"),
                ).where(and_(
                    DailyConsumption.site_id == site_id,
                    DailyConsumption.local_date == local_date,
                ))
            )).one_or_none()
            if row is None or row.hours == 0 or row.total is None:
                return None
            return {"date": local_date, "total": float(row.total),
                    "hours": int(row.hours), "peak_hour": float(row.peak or 0.0)}
        except Exception as e:
            logger.error(f"get_day_consumption({site_id}, {local_date}): {e}")
            return None


async def get_consumption_summary(site_id: int, days: int = 30) -> Dict[str, Any]:
    """Retrieve aggregate consumption statistics for a site over the last `days` days."""
    cache_key = f"summary:{site_id}:{days}"
    cached_data = await cache_get(cache_key)
    if cached_data:
        return cached_data

    async with get_db_session() as session:
        try:
            # Window on the latest recorded timestamp for the site
            latest_ms = await session.scalar(
                select(func.max(DailyConsumption.date))
                .where(DailyConsumption.site_id == site_id)
            )
            if latest_ms is None:
                await cache_set(cache_key, {}, ttl=1800)
                return {}
            cutoff_ms = latest_ms - int(days * 24 * 3600 * 1000)

            daily_totals = (
                select(
                    DailyConsumption.local_date.label("day"),
                    func.sum(DailyConsumption.total_consumption).label("day_total"),
                )
                .where(
                    and_(
                        DailyConsumption.site_id == site_id,
                        DailyConsumption.date >= cutoff_ms,
                        DailyConsumption.date <= latest_ms,
                    )
                )
                .group_by(DailyConsumption.local_date)
                .subquery()
            )

            stmt = select(
                func.sum(daily_totals.c.day_total).label("total"),
                func.avg(daily_totals.c.day_total).label("avg_daily"),
                func.max(daily_totals.c.day_total).label("max_daily"),
                func.count(daily_totals.c.day).label("day_count"),
            )
            result = await session.execute(stmt)
            row = result.fetchone()

            if row and row.day_count and row.day_count > 0:
                data = {
                    "total": float(row.total) if row.total else 0.0,
                    "avg_daily": float(row.avg_daily) if row.avg_daily else 0.0,
                    "max_daily": float(row.max_daily) if row.max_daily else 0.0,
                    "record_count": row.day_count
                }
            else:
                data = {"total": 0.0, "avg_daily": 0.0, "max_daily": 0.0, "record_count": 0}

            # Cache for 2 hours
            await cache_set(cache_key, data, ttl=7200)
            return data
        except Exception as e:
            logger.error(f"Error calculating consumption summary for site {site_id}: {e}")
            return {}

async def save_forecasts(site_id: int, predictions: List[float], model_used: str, domain: str) -> bool:
    """Save forecasted values to the database."""
    async with get_db_session() as session:
        try:
            import time
            current_ms = int(time.time() * 1000)
            # Assuming predictions are hourly, starting from the current hour
            forecast_objects = []
            for i, val in enumerate(predictions):
                date_ms = current_ms + (i * 3600 * 1000)
                forecast = Forecast(
                    site_id=site_id,
                    date=date_ms,
                    predicted_value=float(val),
                    model_used=model_used,
                    domain=domain
                )
                forecast_objects.append(forecast)
            
            session.add_all(forecast_objects)
            await session.commit()
            return True
        except Exception as e:
            logger.error(f"Error saving forecasts for site {site_id}: {e}")
            await session.rollback()
            return False

def describe_anomaly_event(ev: Dict, domain: str) -> str:
    """Generate a human-readable explanation of an anomaly event."""
    unit_word = "electricity" if domain == "energy" else "water"
    unit_label = "kWh" if domain == "energy" else "L"
    duration_h = ev.get("duration_h", 0)
    hour_word = "hour" if duration_h == 1 else "hours"
    base = f"Unusual {unit_word} usage for {duration_h} {hour_word}"

    peak_actual = ev.get("peak_actual")
    peak_expected = ev.get("peak_expected")
    if peak_actual is None or peak_expected is None:
        return f"{base}."

    direction = "higher" if peak_actual > peak_expected else "lower"
    if peak_expected > 0.01:
        pct = abs(peak_actual - peak_expected) / peak_expected * 100
        if pct > 300:  # tiny baseline → % swings wildly; go qualitative
            comparison = f"far {direction} than the ~{peak_expected:.1f} {unit_label} typically expected"
        else:
            comparison = f"about {pct:.0f}% {direction} than the ~{peak_expected:.1f} {unit_label} typically expected"
    else:
        comparison = f"{direction} than the near-zero usage typically expected then"
    return (f"{base} — reached {peak_actual:.1f} {unit_label} "
            f"at its most unusual point, {comparison}.")


async def save_anomalies(device_id: int, events: List[Dict], domain: str) -> bool:
    """Persist detected anomaly events for a device and domain."""
    async with get_db_session() as session:
        try:
            # Replace existing events for this device and domain
            await session.execute(
                delete(Anomaly).where(Anomaly.device_id == device_id, Anomaly.project == domain)
            )

            anomaly_objects = []
            for ev in events:
                start_ms = int(pd.Timestamp(ev["start"]).timestamp() * 1000)
                end_ms   = int(pd.Timestamp(ev["end"]).timestamp() * 1000)
                anomaly = Anomaly(
                    anomaly_name=f"{domain.capitalize()} Anomaly",
                    description=describe_anomaly_event(ev, domain),
                    start_date=start_ms,
                    end_date=end_ms,
                    device_id=device_id,
                    project=domain
                )
                anomaly_objects.append(anomaly)

            session.add_all(anomaly_objects)
            await session.commit()

            # Invalidate cached anomalies and count
            await cache_delete(
                f"anomalies:{device_id}:{domain}:200",
                f"anomalies:{device_id}:{domain}:50",
                f"anomaly_count:{device_id}:{domain}",
            )
            return True
        except Exception as e:
            logger.error(f"Error saving anomalies for device {device_id}: {e}")
            await session.rollback()
            return False

async def save_report(domain: str, report_text: str, source: str,
                       generated_at: Optional[datetime] = None) -> bool:
    """Persist a generated report to the database."""
    async with get_db_session() as session:
        try:
            kwargs = {"domain": domain, "source": source, "report_text": report_text}
            if generated_at is not None:
                kwargs["generated_at"] = generated_at
            session.add(Report(**kwargs))
            await session.commit()
            # Invalidate cached report lists
            await cache_delete("reports:energy", "reports:water")
            return True
        except Exception as e:
            logger.error(f"Error saving {source} report for domain {domain}: {e}")
            await session.rollback()
            return False

async def get_reports(domain: str, limit: int = 30) -> List[Dict[str, Any]]:
    """Retrieve recent reports for a domain, ordered by generation date."""
    cache_key = f"reports:{domain}"
    cached = await cache_get(cache_key)
    if cached is not None:
        return cached

    async with get_db_session() as session:
        try:
            stmt = (
                select(Report.id, Report.domain, Report.source, Report.generated_at, Report.report_text)
                .where(Report.domain.in_([domain, "combined"]))
                .order_by(desc(Report.generated_at))
                .limit(limit)
            )
            result = await session.execute(stmt)
            rows = result.all()
            data = [
                {
                    "id": r.id,
                    "domain": r.domain,
                    "source": r.source,
                    "generated_at": r.generated_at.isoformat() if r.generated_at else None,
                    "report_text": r.report_text,
                }
                for r in rows
            ]
            await cache_set(cache_key, data, ttl=1800)
            return data
        except Exception as e:
            logger.error(f"Error fetching reports for domain {domain}: {e}")
            return []

async def get_report_by_id(report_id: int) -> Optional[Report]:
    """Fetch a single report by ID."""
    async with get_db_session() as session:
        try:
            stmt = select(Report).where(Report.id == report_id)
            result = await session.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error fetching report {report_id}: {e}")
            return None

async def delete_reports(report_ids: List[int]) -> int:
    """Delete stored reports by ID and invalidate report caches."""
    if not report_ids:
        return 0
    async with get_db_session() as session:
        try:
            result = await session.execute(delete(Report).where(Report.id.in_(report_ids)))
            await session.commit()
            await cache_delete("reports:energy", "reports:water")
            return result.rowcount or 0
        except Exception as e:
            logger.error(f"Error deleting reports {report_ids}: {e}")
            await session.rollback()
            return 0

async def get_recent_forecasts(site_id: int, domain: str, limit: int = 24) -> List[Dict[str, Any]]:
    """Retrieve recent forecasts for a site and domain."""
    cache_key = f"forecasts:{site_id}:{domain}:{limit}"
    cached_data = await cache_get(cache_key)
    if cached_data:
        return cached_data

    async with get_db_session() as session:
        try:
            stmt = (
                select(Forecast)
                .where(and_(Forecast.site_id == site_id, Forecast.domain == domain))
                .order_by(desc(Forecast.date))
                .limit(limit)
            )
            result = await session.execute(stmt)
            records = result.scalars().all()
            
            # Return in chronological order
            records.reverse()
            
            data = [
                {
                    "id": r.id,
                    "date": r.date,
                    "predicted_value": r.predicted_value,
                    "model_used": r.model_used
                }
                for r in records
            ]
            
            # Cache for 15 mins (since forecasts change hourly)
            await cache_set(cache_key, data, ttl=900)
            return data
        except Exception as e:
            logger.error(f"Error fetching forecasts for site {site_id}: {e}")
            return []
