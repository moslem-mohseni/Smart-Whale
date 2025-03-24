import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from infrastructure.timescaledb.storage.timescaledb_storage import TimescaleDBStorage
from infrastructure.timescaledb.optimization.cache_manager import CacheManager
from infrastructure.timescaledb.monitoring.metrics_collector import MetricsCollector
from infrastructure.timescaledb.monitoring.slow_query_analyzer import SlowQueryAnalyzer
from infrastructure.timescaledb.monitoring.health_check import HealthCheck

logger = logging.getLogger(__name__)


class DatabaseService:
    """سرویس مدیریت TimescaleDB"""

    def __init__(
        self,
        storage: TimescaleDBStorage,
        cache_manager: CacheManager,
        metrics_collector: MetricsCollector,
        slow_query_analyzer: SlowQueryAnalyzer,
        health_check: HealthCheck
    ):
        """
        مقداردهی اولیه سرویس پایگاه داده

        Args:
            storage (TimescaleDBStorage): مدیریت ذخیره‌سازی
            cache_manager (CacheManager): مدیریت کش کوئری‌ها
            metrics_collector (MetricsCollector): جمع‌آوری متریک‌ها
            slow_query_analyzer (SlowQueryAnalyzer): تحلیل کوئری‌های کند
            health_check (HealthCheck): بررسی سلامت سیستم
        """
        self.storage = storage
        self.cache_manager = cache_manager
        self.metrics_collector = metrics_collector
        self.slow_query_analyzer = slow_query_analyzer
        self.health_check = health_check

    async def execute_query(self, query: str,
                            params: Optional[List[Any]] = None, use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        اجرای کوئری با پشتیبانی از کش

        Args:
            query (str): متن کوئری SQL
            params (Optional[List[Any]]): پارامترهای کوئری
            use_cache (bool): استفاده از کش در صورت امکان

        Returns:
            List[Dict[str, Any]]: نتیجه اجرای کوئری
        """
        cache_key = f"query_cache:{query}:{params}"
        if use_cache:
            cached_result = await self.cache_manager.get_cached_result(cache_key)
            if cached_result:
                logger.info(f"⚡ استفاده از کش برای کوئری: {query}")
                return cached_result

        result = await self.storage.execute_query(query, params)

        if use_cache:
            await self.cache_manager.cache_result(cache_key, result)
        return result

    async def store_time_series_data(self, table: str, id: int, timestamp: datetime,
                                     value: float, metadata: Dict[str, Any]):
        """
        ذخیره داده‌های سری‌زمانی

        Args:
            table (str): نام جدول
            id (int): شناسه
            timestamp (datetime): زمان ثبت داده
            value (float): مقدار عددی داده
            metadata (Dict[str, Any]): اطلاعات متا مرتبط
        """
        query = f"""
            INSERT INTO {table} (id, timestamp, value, metadata)
            VALUES ($1, $2, $3, $4)
        """
        await self.execute_query(query, [id, timestamp, value, metadata], use_cache=False)
        logger.info(f"✅ داده سری‌زمانی ذخیره شد: {id} -> {value}")

    async def get_time_series_data(self, table: str, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """
        دریافت داده‌های سری‌زمانی در بازه مشخص

        Args:
            table (str): نام جدول
            start_time (datetime): زمان شروع بازه
            end_time (datetime): زمان پایان بازه

        Returns:
            List[Dict[str, Any]]: لیست داده‌های سری‌زمانی
        """
        query = f"""
            SELECT * FROM {table}
            WHERE timestamp BETWEEN $1 AND $2
            ORDER BY timestamp ASC
        """
        return await self.execute_query(query, [start_time, end_time])

    async def get_database_metrics(self) -> Dict[str, Any]:
        """
        دریافت متریک‌های پایگاه داده

        Returns:
            Dict[str, Any]: اطلاعات مصرف منابع و سلامت پایگاه داده
        """
        return await self.metrics_collector.get_database_metrics()

    async def get_slow_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        دریافت لیستی از کندترین کوئری‌ها

        Args:
            limit (int): تعداد کوئری‌های کند مورد نظر

        Returns:
            List[Dict[str, Any]]: لیست کوئری‌های کند به همراه جزئیات
        """
        return await self.slow_query_analyzer.get_slow_queries(limit)

    async def check_health(self) -> Dict[str, Any]:
        """
        بررسی وضعیت سلامت پایگاه داده

        Returns:
            Dict[str, Any]: وضعیت کلی سلامت پایگاه داده
        """
        connection_status = await self.health_check.check_connection()
        resource_usage = await self.health_check.get_resource_usage()
        critical_issues = await self.health_check.check_critical_issues()

        return {
            "connection_status": connection_status,
            "resource_usage": resource_usage,
            "critical_issues": critical_issues
        }
