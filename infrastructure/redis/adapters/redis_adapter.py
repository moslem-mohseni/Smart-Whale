import redis.asyncio as redis
import pickle
import logging
import asyncio
from typing import Any, Optional, Dict, List
from ..config.settings import RedisConfig
from .circuit_breaker import CircuitBreaker
from .connection_pool import RedisConnectionPool
from .retry_mechanism import retry_async

logger = logging.getLogger(__name__)


class RedisAdapter:
    """
    مدیریت ارتباط با Redis شامل عملیات CRUD با پشتیبانی از Circuit Breaker و Connection Pooling
    """
    def __init__(self, config: RedisConfig):
        self.config = config
        self._pool = RedisConnectionPool(config)
        self._circuit_breaker = CircuitBreaker()

    async def connect(self):
        """برقراری اتصال به Redis و مقداردهی اولیه Pool"""
        await self._pool.initialize()
        logger.info("Redis connection pool initialized.")

    async def disconnect(self):
        """قطع اتصال از Redis"""
        logger.info("Redis connection closed.")

    @retry_async(retries=3, delay=1)
    async def get(self, key: str) -> Optional[Any]:
        """دریافت مقدار از Redis"""
        async with self._circuit_breaker:
            connection = await self._pool.acquire()
            try:
                value = await connection.get(key)
                return pickle.loads(value) if value else None
            finally:
                await self._pool.release(connection)

    @retry_async(retries=3, delay=1)
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """ذخیره مقدار در Redis"""
        async with self._circuit_breaker:
            connection = await self._pool.acquire()
            try:
                serialized = pickle.dumps(value)
                if ttl:
                    await connection.setex(key, ttl, serialized)
                else:
                    await connection.set(key, serialized)
            finally:
                await self._pool.release(connection)

    @retry_async(retries=3, delay=1)
    async def delete(self, key: str) -> bool:
        """حذف مقدار از Redis"""
        async with self._circuit_breaker:
            connection = await self._pool.acquire()
            try:
                return await connection.delete(key) > 0
            finally:
                await self._pool.release(connection)

    @retry_async(retries=3, delay=1)
    async def flush(self) -> None:
        """پاک‌سازی کل کش"""
        async with self._circuit_breaker:
            connection = await self._pool.acquire()
            try:
                await connection.flushdb()
                logger.info("Redis cache flushed.")
            finally:
                await self._pool.release(connection)

    @retry_async(retries=3, delay=1)
    async def hset(self, key: str, field: str, value: Any) -> None:
        """ذخیره مقدار در HashMap"""
        async with self._circuit_breaker:
            connection = await self._pool.acquire()
            try:
                await connection.hset(key, field, pickle.dumps(value))
            finally:
                await self._pool.release(connection)

    @retry_async(retries=3, delay=1)
    async def hget(self, key: str, field: str) -> Optional[Any]:
        """دریافت مقدار از HashMap"""
        async with self._circuit_breaker:
            connection = await self._pool.acquire()
            try:
                value = await connection.hget(key, field)
                return pickle.loads(value) if value else None
            finally:
                await self._pool.release(connection)

    @retry_async(retries=3, delay=1)
    async def expire(self, key: str, ttl: int) -> bool:
        """تنظیم زمان انقضا برای کلید"""
        async with self._circuit_breaker:
            connection = await self._pool.acquire()
            try:
                return await connection.expire(key, ttl)
            finally:
                await self._pool.release(connection)

    @retry_async(retries=3, delay=1)
    async def incr(self, key: str, amount: int = 1) -> int:
        """افزایش مقدار عددی یک کلید"""
        async with self._circuit_breaker:
            connection = await self._pool.acquire()
            try:
                return await connection.incrby(key, amount)
            finally:
                await self._pool.release(connection)
