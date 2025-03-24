import logging
from typing import List, Dict, Any
from infrastructure.timescaledb.storage.timescaledb_storage import TimescaleDBStorage

logger = logging.getLogger(__name__)


class SlowQueryAnalyzer:
    """تحلیل و بررسی کوئری‌های کند در TimescaleDB"""

    def __init__(self, storage: TimescaleDBStorage):
        """
        مقداردهی اولیه

        Args:
            storage (TimescaleDBStorage): شیء ذخیره‌سازی برای اجرای کوئری‌ها
        """
        self.storage = storage

    async def get_slow_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        دریافت لیستی از کندترین کوئری‌ها

        Args:
            limit (int): تعداد کوئری‌های کند مورد نظر

        Returns:
            List[Dict[str, Any]]: لیست کوئری‌های کند به همراه جزئیات
        """
        query = f"""
            SELECT 
                query,
                calls,
                total_time,
                mean_time,
                rows,
                shared_blks_hit,
                shared_blks_read,
                shared_blks_written,
                temp_blks_written
            FROM pg_stat_statements
            ORDER BY mean_time DESC
            LIMIT {limit};
        """

        try:
            logger.info(f"🔍 استخراج {limit} کوئری کند برتر از `pg_stat_statements`...")
            result = await self.storage.execute_query(query)
            logger.info("✅ تحلیل کوئری‌های کند انجام شد.")
            return result
        except Exception as e:
            logger.error(f"❌ خطا در تحلیل کوئری‌های کند: {e}")
            return []

    async def reset_query_stats(self):
        """
        بازنشانی آمار `pg_stat_statements` برای شروع تحلیل جدید
        """
        query = "SELECT pg_stat_statements_reset();"

        try:
            logger.info("♻️ بازنشانی آمار `pg_stat_statements`...")
            await self.storage.execute_query(query)
            logger.info("✅ آمار کوئری‌ها بازنشانی شد.")
        except Exception as e:
            logger.error(f"❌ خطا در بازنشانی آمار کوئری‌ها: {e}")
            raise
