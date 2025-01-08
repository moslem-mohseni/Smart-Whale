# infrastructure/clickhouse/service/analytics_service.py

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from ..adapters.clickhouse_adapter import ClickHouseAdapter
from ..config.settings import ClickHouseConfig, QuerySettings
from ..domain.models import (
    AnalyticsEvent,
    AnalyticsQuery,
    AnalyticsResult,
    TableSchema
)
from .analytics_cache import AnalyticsCache
from ...redis.config.settings import RedisConfig
from ...interfaces.exceptions import ConnectionError, OperationError

logger = logging.getLogger(__name__)


class AnalyticsService:
    """سرویس تحلیلی برای کار با داده‌های ClickHouse"""

    def __init__(self, config: ClickHouseConfig, redis_config: RedisConfig):
        """راه‌اندازی سرویس با تنظیمات"""
        self.config = config
        self._adapter = ClickHouseAdapter(config)
        self._cache = AnalyticsCache(redis_config)
        self._initialized_tables = set()

    async def connect(self) -> None:
        """برقراری اتصال به سرویس‌ها"""
        await self._adapter.connect()
        await self._cache.initialize()

    async def initialize(self) -> None:
        """راه‌اندازی اولیه سرویس و ایجاد جداول پایه"""
        await self.connect()
        await self._ensure_base_tables()
        logger.info("Analytics service initialized successfully")

    async def _ensure_base_tables(self) -> None:
        """ایجاد جداول پایه مورد نیاز"""
        events_schema = TableSchema(
            name='events',
            columns={
                'event_id': 'String',
                'event_type': 'LowCardinality(String)',
                'timestamp': 'DateTime',
                'data': 'String',  # JSON
                'metadata': 'String'  # JSON
            },
            engine='MergeTree()',
            partition_by='toYYYYMM(timestamp)',
            order_by=['timestamp', 'event_type', 'event_id']
        )

        if 'events' not in self._initialized_tables:
            await self._adapter.create_table('events', events_schema.columns)
            self._initialized_tables.add('events')
            logger.info("Created events table")

    async def store_event(self, event: AnalyticsEvent) -> None:
        """ذخیره یک رویداد تحلیلی"""
        query = """
            INSERT INTO events (event_id, event_type, timestamp, data, metadata)
            VALUES ($1, $2, $3, $4, $5)
        """
        params = [
            event.event_id,
            event.event_type,
            event.timestamp,
            str(event.data),
            str(event.metadata) if event.metadata else None
        ]
        await self._adapter.execute(query, params)
        logger.debug(f"Stored event: {event.event_id}")

    async def execute_analytics_query(self, query: AnalyticsQuery) -> AnalyticsResult:
        """
        اجرای یک پرس‌وجوی تحلیلی با پشتیبانی از کش و مدیریت خطا

        Args:
            query: پرس‌وجوی تحلیلی

        Returns:
            نتیجه تحلیل

        Raises:
            ConnectionError: در صورت بروز مشکل در اتصال به دیتابیس
            OperationError: در صورت بروز خطا در اجرای پرس‌وجو
        """
        # بررسی کش
        try:
            cached_result = await self._cache.get_cached_result(query)
            if cached_result:
                logger.debug("Query result found in cache")
                return cached_result
        except Exception as e:
            logger.warning(f"Cache lookup failed: {str(e)}")
            # در صورت خطا در کش، ادامه می‌دهیم تا از دیتابیس بخوانیم

        # اجرای پرس‌وجو
        try:
            start_time = datetime.now()
            result = await self._adapter.execute_analytics_query(query)
            execution_time = (datetime.now() - start_time).total_seconds()

        except ConnectionError:
            # خطای اتصال را مستقیماً منتقل می‌کنیم
            logger.error("Database connection failed", exc_info=True)
            raise

        except Exception as e:
            # سایر خطاها را به OperationError تبدیل می‌کنیم
            error_msg = f"Query execution failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise OperationError(error_msg) from e

        # ایجاد نتیجه
        analytics_result = AnalyticsResult(
            query=query,
            data=result,
            total_count=len(result),
            execution_time=execution_time
        )

        # ذخیره در کش (خطاهای کش را نادیده می‌گیریم)
        try:
            await self._cache.cache_result(analytics_result)
        except Exception as e:
            logger.warning(f"Failed to cache result: {str(e)}")

        return analytics_result

    async def get_event_trends(self,
                               event_type: str,
                               interval: str = '1 hour',
                               days: int = 7,
                               use_cache: bool = True) -> AnalyticsResult:
        """دریافت روند رویدادها در طول زمان"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        query = AnalyticsQuery(
            dimensions=[f"toStartOf{interval}(timestamp) as period"],
            metrics=['count() as count'],
            filters={'event_type': event_type},
            time_range=(start_time, end_time),
            order_by=['period']
        )

        if not use_cache:
            result = await self._adapter.execute_analytics_query(query)
            return AnalyticsResult(
                query=query,
                data=result,
                total_count=len(result),
                execution_time=0
            )

        return await self.execute_analytics_query(query)

    async def invalidate_event_cache(self, event_type: Optional[str] = None) -> None:
        """حذف کش برای یک نوع رویداد خاص یا تمام رویدادها"""
        if event_type:
            pattern = f"query:*{event_type}*"
        else:
            pattern = None
        await self._cache.invalidate_cache(pattern)

    async def get_cache_stats(self) -> Dict[str, Any]:
        """دریافت آمار کش"""
        return await self._cache.get_cache_stats()

    async def shutdown(self) -> None:
        """خاتمه سرویس"""
        await self._adapter.disconnect()
        await self._cache.shutdown()
        logger.info("Analytics service shut down successfully")