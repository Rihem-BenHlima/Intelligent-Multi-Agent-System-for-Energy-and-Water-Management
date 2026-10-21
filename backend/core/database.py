"""
core/database.py
────────────────
SQLAlchemy async engine + session factory for MySQL (aiomysql driver).
"""
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from core.config import settings

engine = create_async_engine(
    settings.database_url,
    pool_pre_ping=False,
    pool_size=5,
    max_overflow=10,
    echo=False,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


async def init_db() -> None:
    """Create all tables on startup."""
    from models import db_models  # noqa: F401 — registers ORM models
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await _add_missing_columns(conn)


async def _add_missing_columns(conn) -> None:
    """Idempotently add new columns to existing tables if missing."""
    result = await conn.execute(text("""
        SELECT COUNT(*) FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = 'anomalies'
          AND COLUMN_NAME = 'anomaly_type'
    """))
    if not result.scalar():
        await conn.execute(text(
            "ALTER TABLE anomalies ADD COLUMN anomaly_type VARCHAR(20) NULL"
        ))
