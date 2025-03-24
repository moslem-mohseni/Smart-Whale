import asyncio
import redis.asyncio as redis
from typing import Optional, List
from ..config.settings import RedisConfig


class RedisConnectionPool:
    """
    مدیریت Connection Pool برای Redis جهت بهبود کارایی و مدیریت اتصال‌های همزمان
    """

    def __init__(self, config: RedisConfig, min_size: int = 5, max_size: int = 10):
        self.config = config
        self.min_size = min_size
        self.max_size = max_size
        self._pool: List[redis.Redis] = []
        self._in_use = set()
        self._lock = asyncio.Lock()

    async def initialize(self):
        """ایجاد اتصالات اولیه برای Connection Pool"""
        for _ in range(self.min_size):
            connection = await self._create_connection()
            self._pool.append(connection)

    async def _create_connection(self) -> redis.Redis:
        """ایجاد یک اتصال جدید به Redis"""
        return redis.from_url(
            f"redis://{self.config.host}:{self.config.port}",
            db=self.config.database,
            password=self.config.password,
            max_connections=self.max_size
        )

    async def acquire(self) -> redis.Redis:
        """دریافت یک اتصال از Pool"""
        async with self._lock:
            while len(self._pool) == 0:
                if len(self._in_use) < self.max_size:
                    connection = await self._create_connection()
                    self._in_use.add(connection)
                    return connection
                await asyncio.sleep(0.1)

            connection = self._pool.pop()
            self._in_use.add(connection)
            return connection

    async def release(self, connection: redis.Redis):
        """بازگرداندن اتصال به Pool"""
        async with self._lock:
            self._in_use.remove(connection)
            if len(self._pool) < self.min_size:
                self._pool.append(connection)
            else:
                await connection.close()



