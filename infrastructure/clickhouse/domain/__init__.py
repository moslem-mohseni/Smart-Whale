# infrastructure/clickhouse/domain/__init__.py
"""
این ماژول شامل مدل‌های دامنه برای کار با ClickHouse است.
ساختارهای داده و کلاس‌های مرتبط با منطق کسب‌وکار در این بخش تعریف می‌شوند.
"""

from .models import (
    AnalyticsEvent,
    AnalyticsQuery,
    AnalyticsResult,
    TableSchema
)

__all__ = [
    'AnalyticsEvent',
    'AnalyticsQuery',
    'AnalyticsResult',
    'TableSchema'
]
