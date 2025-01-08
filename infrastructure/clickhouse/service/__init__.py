# infrastructure/clickhouse/service/__init__.py
"""
این ماژول سرویس‌های سطح بالا برای کار با ClickHouse را فراهم می‌کند.
این سرویس‌ها یک رابط ساده و یکپارچه برای انجام عملیات تحلیلی ارائه می‌دهند.
"""

from .analytics_service import AnalyticsService
from .analytics_cache import AnalyticsCache

__all__ = ['AnalyticsService','AnalyticsCache']