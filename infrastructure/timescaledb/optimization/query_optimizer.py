import logging
from typing import List, Dict, Any, Optional
from infrastructure.timescaledb.storage.timescaledb_storage import TimescaleDBStorage
from infrastructure.timescaledb.optimization.cache_manager import CacheManager

logger = logging.getLogger(__name__)


class QueryOptimizer:
    """تحلیل و بهینه‌سازی کوئری‌های TimescaleDB"""

    def __init__(self, storage: TimescaleDBStorage, cache_manager: CacheManager):
        """
        مقداردهی اولیه

        Args:
            storage (TimescaleDBStorage): شیء ذخیره‌سازی برای اجرای کوئری‌ها
            cache_manager (CacheManager): مدیریت کش برای ذخیره تحلیل‌های قبلی
        """
        self.storage = storage
        self.cache_manager = cache_manager

    async def analyze_query(self, query: str, params: Optional[List[Any]] = None) -> Dict[str, Any]:
        """
        تحلیل کوئری با `EXPLAIN ANALYZE`

        Args:
            query (str): متن کوئری SQL
            params (Optional[List[Any]]): پارامترهای کوئری (در صورت نیاز)

        Returns:
            Dict[str, Any]: نتایج تحلیل کوئری
        """
        cache_key = f"query_analysis:{query}"
        cached_result = await self.cache_manager.get_cached_result(cache_key)

        if cached_result:
            logger.info(f"⚡ تحلیل کش شده برای کوئری `{query}` بازیابی شد.")
            return cached_result

        analyze_query = f"EXPLAIN ANALYZE {query}"
        result = await self.storage.execute_query(analyze_query, params)

        analysis = {"query": query, "analysis": result}
        await self.cache_manager.cache_result(cache_key, analysis, ttl=600)

        logger.info(f"📊 تحلیل جدید برای کوئری `{query}` انجام شد و کش شد.")
        return analysis

    async def suggest_indexes(self, table: str) -> List[str]:
        """
        پیشنهاد ایندکس‌های مناسب بر اساس تحلیل جدول

        Args:
            table (str): نام جدول

        Returns:
            List[str]: لیستی از پیشنهادات ایندکس
        """
        cache_key = f"index_suggestions:{table}"
        cached_indexes = await self.cache_manager.get_cached_result(cache_key)

        if cached_indexes:
            logger.info(f"⚡ پیشنهادات کش شده ایندکس برای `{table}` بازیابی شد.")
            return cached_indexes

        index_suggestions = []
        query = f"""
            SELECT attname AS column_name, null_frac, n_distinct
            FROM pg_stats
            WHERE tablename = '{table}';
        """
        stats = await self.storage.execute_query(query)

        for stat in stats:
            column = stat["column_name"]
            null_frac = stat["null_frac"]
            n_distinct = stat["n_distinct"]

            if null_frac > 0.3:  # اگر درصد مقادیر Null زیاد باشد
                index_suggestions.append(f"CREATE INDEX IF NOT EXISTS idx_{table}_{column} ON {table} ({column}) WHERE {column} IS NOT NULL;")
            elif abs(n_distinct) > 1000:  # اگر تعداد مقادیر یکتا زیاد باشد
                index_suggestions.append(f"CREATE INDEX IF NOT EXISTS idx_{table}_{column} ON {table} ({column});")

        await self.cache_manager.cache_result(cache_key, index_suggestions, ttl=3600)
        logger.info(f"📌 پیشنهادات ایندکس برای `{table}` محاسبه و کش شد.")
        return index_suggestions
