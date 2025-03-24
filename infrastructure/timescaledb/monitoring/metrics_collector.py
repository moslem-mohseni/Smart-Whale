import logging
from typing import Dict, Any
from infrastructure.timescaledb.storage.timescaledb_storage import TimescaleDBStorage

logger = logging.getLogger(__name__)


class MetricsCollector:
    """جمع‌آوری متریک‌های عملکردی TimescaleDB"""

    def __init__(self, storage: TimescaleDBStorage):
        """
        مقداردهی اولیه

        Args:
            storage (TimescaleDBStorage): شیء ذخیره‌سازی برای اجرای کوئری‌ها
        """
        self.storage = storage

    async def get_database_metrics(self) -> Dict[str, Any]:
        """
        استخراج اطلاعات کلی از `pg_stat_database`

        Returns:
            Dict[str, Any]: متریک‌های پایگاه داده
        """
        query = """
            SELECT 
                datname AS database_name,
                numbackends AS active_connections,
                xact_commit AS committed_transactions,
                xact_rollback AS rolledback_transactions,
                blks_read AS blocks_read,
                blks_hit AS cache_hits,
                deadlocks,
                temp_files,
                temp_bytes
            FROM pg_stat_database
            WHERE datname = current_database();
        """

        try:
            result = await self.storage.execute_query(query)
            return result[0] if result else {}
        except Exception as e:
            logger.error(f"❌ خطا در دریافت متریک‌های پایگاه داده: {e}")
            return {}

    async def get_bgwriter_metrics(self) -> Dict[str, Any]:
        """
        استخراج اطلاعات `pg_stat_bgwriter` برای مانیتورینگ مصرف منابع

        Returns:
            Dict[str, Any]: متریک‌های مربوط به Background Writer
        """
        query = """
            SELECT 
                checkpoints_timed,
                checkpoints_req,
                buffers_checkpoint,
                buffers_clean,
                buffers_backend
            FROM pg_stat_bgwriter;
        """

        try:
            result = await self.storage.execute_query(query)
            return result[0] if result else {}
        except Exception as e:
            logger.error(f"❌ خطا در دریافت متریک‌های Background Writer: {e}")
            return {}

    async def get_database_size(self) -> int:
        """
        دریافت اندازه دیتابیس در بایت

        Returns:
            int: اندازه دیتابیس بر حسب بایت
        """
        query = "SELECT pg_database_size(current_database()) AS size;"

        try:
            result = await self.storage.execute_query(query)
            return result[0]["size"] if result else 0
        except Exception as e:
            logger.error(f"❌ خطا در دریافت اندازه دیتابیس: {e}")
            return 0
