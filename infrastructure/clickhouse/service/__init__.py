# infrastructure/clickhouse/service/__init__.py
"""
ماژول سرویس‌های ClickHouse

این ماژول شامل سرویس‌های سطح بالا برای ClickHouse است:
- AnalyticsCache: مدیریت کش برای نتایج تحلیلی
- AnalyticsService: سرویس تحلیل داده‌ها با پشتیبانی از کش و بهینه‌سازی

برای استفاده از این ماژول، توصیه می‌شود از factory method‌ها استفاده کنید
که تنظیمات مناسب را به‌صورت خودکار تنظیم می‌کنند.

مثال:
    ```python
    from infrastructure.clickhouse.service import create_analytics_service

    # ایجاد سرویس با تنظیمات پیش‌فرض
    analytics_service = create_analytics_service()

    # استفاده از سرویس
    result = await analytics_service.execute_analytics_query(query)
    ```
"""

import logging
from typing import Optional
from ..config.config import config
from ..adapters.clickhouse_adapter import ClickHouseAdapter
from ..optimization.query_optimizer import QueryOptimizer
from .analytics_cache import AnalyticsCache
from .analytics_service import AnalyticsService

logger = logging.getLogger(__name__)

logger.info("Initializing ClickHouse Service Module...")

__all__ = [
    "AnalyticsCache",
    "AnalyticsService",
    "create_analytics_service",
    "create_analytics_cache"
]


def create_analytics_cache() -> AnalyticsCache:
    """
    ایجاد یک نمونه از AnalyticsCache با تنظیمات مناسب

    Returns:
        AnalyticsCache: نمونه آماده استفاده از AnalyticsCache
    """
    # نکته: مدیریت تنظیمات Redis در ماژول Redis انجام می‌شود
    from ...redis.config.settings import RedisConfig
    redis_config = RedisConfig()
    return AnalyticsCache(redis_config=redis_config)


def create_analytics_service(
        clickhouse_adapter: Optional[ClickHouseAdapter] = None,
        use_analytics_cache: bool = True
) -> AnalyticsService:
    """
    ایجاد یک نمونه از AnalyticsService با تنظیمات مناسب

    این تابع تمام وابستگی‌های مورد نیاز برای AnalyticsService را ایجاد
    و پیکربندی می‌کند.

    Args:
        clickhouse_adapter (ClickHouseAdapter, optional): آداپتور ClickHouse سفارشی
        use_analytics_cache (bool): استفاده از AnalyticsCache

    Returns:
        AnalyticsService: نمونه آماده استفاده از AnalyticsService
    """
    # استفاده از آداپتور موجود یا ایجاد آداپتور جدید
    adapter = clickhouse_adapter or ClickHouseAdapter(config)

    # ایجاد optimizer
    optimizer = QueryOptimizer(adapter)

    # مقداردهی cache
    analytics_cache = None
    cache_manager = None

    if use_analytics_cache:
        # استفاده از AnalyticsCache
        analytics_cache = create_analytics_cache()
    else:
        # تلاش برای ایجاد CacheManager
        try:
            # این بخش در صورت نیاز به CacheManager پیاده‌سازی می‌شود
            # در حال حاضر، فقط متغیر را None قرار می‌دهیم
            logger.info("CacheManager integration not implemented, using AnalyticsCache instead")
            analytics_cache = create_analytics_cache()
        except Exception as e:
            logger.warning(f"Failed to create cache, proceeding without cache: {str(e)}")

    # ایجاد و برگرداندن سرویس
    return AnalyticsService(
        clickhouse_adapter=adapter,
        query_optimizer=optimizer,
        analytics_cache=analytics_cache
    )