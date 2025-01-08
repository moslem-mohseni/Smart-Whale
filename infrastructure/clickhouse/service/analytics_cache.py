# infrastructure/clickhouse/service/analytics_cache.py

import logging
import json
import hashlib
from typing import Optional, Any, Dict
from datetime import datetime, timedelta
from dataclasses import asdict

from ..domain.models import AnalyticsQuery, AnalyticsResult
from ...redis import CacheService
from ...redis.config.settings import RedisConfig
from ...redis.domain.models import CacheNamespace

logger = logging.getLogger(__name__)


class AnalyticsCache:
    """
    سیستم کش‌گذاری برای نتایج تحلیلی

    این کلاس مسئول کش کردن نتایج پرس‌وجوهای تحلیلی پرتکرار است و
    از Redis برای ذخیره‌سازی استفاده می‌کند.
    """

    def __init__(self, redis_config: RedisConfig):
        """
        راه‌اندازی سیستم کش

        Args:
            redis_config: تنظیمات اتصال به Redis
        """
        self.cache_service = CacheService(redis_config)
        self.namespace = CacheNamespace(
            name="analytics",
            default_ttl=3600,  # یک ساعت پیش‌فرض
            max_size=1000000  # حداکثر یک میلیون کلید
        )

    async def initialize(self) -> None:
        """راه‌اندازی اولیه کش"""
        await self.cache_service.connect()
        self.cache_service.create_namespace(self.namespace)
        logger.info("Analytics cache initialized")

    async def shutdown(self) -> None:
        """خاتمه سرویس کش"""
        await self.cache_service.disconnect()
        logger.info("Analytics cache shut down")

    def _generate_cache_key(self, query: AnalyticsQuery) -> str:
        """
        تولید کلید یکتا برای پرس‌وجو

        Args:
            query: پرس‌وجوی تحلیلی

        Returns:
            کلید یکتا برای کش
        """
        # تبدیل پرس‌وجو به دیکشنری
        query_dict = {
            'dimensions': sorted(query.dimensions),
            'metrics': sorted(query.metrics),
            'filters': json.dumps(query.filters, sort_keys=True) if query.filters else None,
            'time_range': [dt.isoformat() for dt in query.time_range] if query.time_range else None,
            'limit': query.limit,
            'order_by': sorted(query.order_by) if query.order_by else None
        }

        # تولید کلید با استفاده از هش
        query_str = json.dumps(query_dict, sort_keys=True)
        return f"query:{hashlib.sha256(query_str.encode()).hexdigest()}"

    async def get_cached_result(self, query: AnalyticsQuery) -> Optional[AnalyticsResult]:
        """
        بازیابی نتیجه از کش

        Args:
            query: پرس‌وجوی تحلیلی

        Returns:
            نتیجه کش شده یا None
        """
        try:
            cache_key = self._generate_cache_key(query)
            cached_data = await self.cache_service.get(cache_key, namespace=self.namespace.name)

            if cached_data:
                return AnalyticsResult(
                    query=query,
                    data=cached_data['data'],
                    total_count=cached_data['total_count'],
                    execution_time=cached_data['execution_time'],
                    metadata={'from_cache': True}
                )

            return None

        except Exception as e:
            logger.warning(f"Error retrieving from cache: {str(e)}")
            return None

    async def cache_result(self, result: AnalyticsResult, ttl: Optional[int] = None) -> None:
        """
        ذخیره نتیجه در کش

        Args:
            result: نتیجه تحلیلی
            ttl: زمان انقضا به ثانیه (اختیاری)
        """
        try:
            cache_key = self._generate_cache_key(result.query)
            cache_data = {
                'data': result.data,
                'total_count': result.total_count,
                'execution_time': result.execution_time,
                'cached_at': datetime.now().isoformat()
            }

            await self.cache_service.set(
                key=cache_key,
                value=cache_data,
                ttl=ttl,
                namespace=self.namespace.name
            )
            logger.debug(f"Cached result for key: {cache_key}")

        except Exception as e:
            logger.warning(f"Error caching result: {str(e)}")

    async def invalidate_cache(self, pattern: Optional[str] = None) -> None:
        """
        حذف داده‌های کش

        Args:
            pattern: الگوی کلیدهای مورد نظر برای حذف (اختیاری)
        """
        try:
            if pattern:
                keys = await self.cache_service.scan_keys(
                    pattern,
                    namespace=self.namespace.name
                )
                for key in keys:
                    await self.cache_service.delete(key, namespace=self.namespace.name)
            else:
                # حذف کل namespace
                await self.cache_service.delete_namespace(self.namespace.name)

            logger.info(f"Invalidated cache with pattern: {pattern or 'all'}")

        except Exception as e:
            logger.error(f"Error invalidating cache: {str(e)}")

    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        دریافت آمار کش

        Returns:
            دیکشنری حاوی آمار کش
        """
        try:
            keys = await self.cache_service.scan_keys(
                "*",
                namespace=self.namespace.name
            )

            total_keys = len(keys)
            memory_used = 0
            expired_keys = 0

            for key in keys:
                ttl = await self.cache_service.ttl(key, namespace=self.namespace.name)
                if ttl == 0:
                    expired_keys += 1

                # محاسبه تقریبی حجم اشغال شده
                value = await self.cache_service.get(key, namespace=self.namespace.name)
                if value:
                    memory_used += len(str(value))

            return {
                'total_keys': total_keys,
                'expired_keys': expired_keys,
                'memory_used_bytes': memory_used,
                'hit_rate': await self.cache_service.get_hit_rate(self.namespace.name)
            }

        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
            return {}