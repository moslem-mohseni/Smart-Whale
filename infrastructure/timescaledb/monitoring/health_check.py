import logging
from typing import Dict, Any
from infrastructure.timescaledb.storage.timescaledb_storage import TimescaleDBStorage

logger = logging.getLogger(__name__)


class HealthCheck:
    """بررسی سلامت TimescaleDB"""

    def __init__(self, storage: TimescaleDBStorage):
        """
        مقداردهی اولیه

        Args:
            storage (TimescaleDBStorage): شیء ذخیره‌سازی برای اجرای کوئری‌ها
        """
        self.storage = storage

    async def check_connection(self) -> bool:
        """
        بررسی وضعیت اتصال به پایگاه داده

        Returns:
            bool: True اگر اتصال برقرار باشد، False در غیر این صورت
        """
        query = "SELECT 1;"
        try:
            result = await self.storage.execute_query(query)
            return result is not None
        except Exception as e:
            logger.error(f"❌ خطا در بررسی اتصال پایگاه داده: {e}")
            return False

    async def get_resource_usage(self) -> Dict[str, Any]:
        """
        بررسی میزان استفاده از منابع پایگاه داده

        Returns:
            Dict[str, Any]: اطلاعات مصرف منابع پایگاه داده
        """
        query = """
            SELECT 
                numbackends AS active_connections,
                blks_read AS disk_reads,
                blks_hit AS cache_hits,
                deadlocks
            FROM pg_stat_database
            WHERE datname = current_database();
        """

        try:
            logger.info("📊 بررسی مصرف منابع پایگاه داده...")
            result = await self.storage.execute_query(query)
            return result[0] if result else {}
        except Exception as e:
            logger.error(f"❌ خطا در بررسی مصرف منابع: {e}")
            return {}

    async def check_critical_issues(self) -> Dict[str, Any]:
        """
        بررسی مشکلات بحرانی مانند Deadlock و تأخیر در اجرای تراکنش‌ها

        Returns:
            Dict[str, Any]: اطلاعات مربوط به مشکلات بحرانی پایگاه داده
        """
        query = """
            SELECT 
                waiting.pid AS waiting_pid,
                blocking.pid AS blocking_pid,
                waiting.query AS waiting_query,
                blocking.query AS blocking_query
            FROM pg_stat_activity waiting
            JOIN pg_stat_activity blocking
            ON waiting.wait_event IS NOT NULL
            WHERE waiting.pid != blocking.pid;
        """

        try:
            logger.info("🚨 بررسی مشکلات بحرانی پایگاه داده...")
            result = await self.storage.execute_query(query)
            return result if result else {}
        except Exception as e:
            logger.error(f"❌ خطا در بررسی مشکلات بحرانی پایگاه داده: {e}")
            return {}
