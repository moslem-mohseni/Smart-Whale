import asyncpg
import logging
from typing import Optional
from .settings import TimescaleDBConfig

logger = logging.getLogger(__name__)


class ConnectionPool:
    """مدیریت Connection Pool برای TimescaleDB"""

    def __init__(self, config: TimescaleDBConfig):
        """
        مقداردهی اولیه Connection Pool

        Args:
            config (TimescaleDBConfig): تنظیمات پایگاه داده
        """
        self.config = config
        self._pool: Optional[asyncpg.Pool] = None
        self._read_pool: Optional[asyncpg.Pool] = None  # برای Read Replica (در صورت وجود)

    async def initialize(self):
        """ایجاد Connection Pool برای خواندن و نوشتن"""
        try:
            self._pool = await asyncpg.create_pool(
                dsn=self.config.get_connection_string(),
                min_size=self.config.min_connections,
                max_size=self.config.max_connections,
                command_timeout=self.config.connection_timeout,
            )
            logger.info("✅ Connection pool برای TimescaleDB ایجاد شد.")

            # ایجاد Connection Pool برای Read Replica (در صورت وجود)
            if self.config.read_replica:
                self._read_pool = await asyncpg.create_pool(
                    dsn=self.config.get_read_replica_connection(),
                    min_size=self.config.min_connections,
                    max_size=self.config.max_connections,
                    command_timeout=self.config.connection_timeout,
                )
                logger.info("✅ Read Replica Connection Pool ایجاد شد.")

        except Exception as e:
            logger.error(f"❌ خطا در ایجاد Connection Pool: {e}")
            raise

    async def get_connection(self, read_only: bool = False) -> asyncpg.Connection:
        """
        دریافت یک اتصال از Pool

        Args:
            read_only (bool): اگر True باشد، اتصال از Read Replica گرفته می‌شود (در صورت وجود)

        Returns:
            asyncpg.Connection: اتصال به پایگاه داده
        """
        if read_only and self._read_pool:
            return await self._read_pool.acquire()
        return await self._pool.acquire()

    async def release_connection(self, conn: asyncpg.Connection, read_only: bool = False):
        """
        آزاد کردن اتصال به Pool

        Args:
            conn (asyncpg.Connection): اتصال باز
            read_only (bool): مشخص می‌کند که این اتصال متعلق به Read Replica است یا نه
        """
        if read_only and self._read_pool:
            await self._read_pool.release(conn)
        else:
            await self._pool.release(conn)

    async def close(self):
        """بستن تمام اتصال‌ها و آزاد کردن منابع"""
        if self._pool:
            await self._pool.close()
            logger.info("✅ Connection Pool اصلی بسته شد.")
        if self._read_pool:
            await self._read_pool.close()
            logger.info("✅ Read Replica Connection Pool بسته شد.")
