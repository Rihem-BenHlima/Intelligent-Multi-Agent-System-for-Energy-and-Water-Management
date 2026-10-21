"""
core/redis_client.py
────────────────────
Async Redis connection pool + simple get/set cache helpers.
"""
import json
from typing import Any, Optional

import redis.asyncio as aioredis
from core.config import settings

_pool: Optional[aioredis.ConnectionPool] = None

async def init_redis_pool():
    global _pool
    if _pool is None:
        _pool = aioredis.ConnectionPool.from_url(settings.REDIS_URL)

async def get_redis_cache() -> aioredis.Redis:
    """Returns a Redis client configured for general app caching (decodes to strings)."""
    if _pool is None:
        await init_redis_pool()
    return aioredis.Redis(connection_pool=_pool, decode_responses=True)

async def get_redis_checkpointer() -> aioredis.Redis:
    """Returns a Redis client configured for LangGraph checkpointer (raw bytes)."""
    if _pool is None:
        await init_redis_pool()
    return aioredis.Redis(connection_pool=_pool, decode_responses=False)


async def cache_get(key: str) -> Optional[Any]:
    r = await get_redis_cache()
    raw = await r.get(key)
    if raw is None:
        return None
    return json.loads(raw)


async def cache_set(key: str, value: Any, ttl: int = 3600) -> None:
    r = await get_redis_cache()
    await r.setex(key, ttl, json.dumps(value))


async def cache_delete(*keys: str) -> None:
    """Invalidate one or more cache keys immediately."""
    if not keys:
        return
    r = await get_redis_cache()
    await r.delete(*keys)


async def close_redis() -> None:
    global _pool
    if _pool:
        await _pool.disconnect()
        _pool = None
