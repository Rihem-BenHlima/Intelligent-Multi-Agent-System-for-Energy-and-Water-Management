"""FastAPI dependency injection helpers."""
from sqlalchemy.ext.asyncio import AsyncSession
from core.database import AsyncSessionLocal


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
