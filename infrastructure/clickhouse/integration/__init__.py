# infrastructure/clickhouse/integration/__init__.py
"""
ماژول ادغام ClickHouse

این ماژول شامل کلاس‌ها و توابع مربوط به ادغام ClickHouse با سایر سیستم‌ها است:
- GraphQLLayer: لایه GraphQL برای اجرای کوئری‌های تحلیلی
- RestAPI: API REST برای دسترسی به داده‌های ClickHouse
- StreamProcessor: پردازش داده‌های استریم و درج در ClickHouse
"""

import logging
from .graphql_layer import GraphQLLayer
from .rest_api import RestAPI
from .stream_processor import StreamProcessor
from ..service.analytics_service import AnalyticsService
from ..adapters.clickhouse_adapter import ClickHouseAdapter

logger = logging.getLogger(__name__)

logger.info("Initializing ClickHouse Integration Module...")

__all__ = [
    "GraphQLLayer",
    "RestAPI",
    "StreamProcessor",
    "create_graphql_layer",
    "create_rest_api",
    "create_stream_processor"
]


def create_graphql_layer(analytics_service: AnalyticsService = None):
    """
    ایجاد یک نمونه از GraphQLLayer با تنظیمات مناسب

    Args:
        analytics_service (AnalyticsService, optional): سرویس تحلیل داده‌ها

    Returns:
        GraphQLLayer: نمونه آماده استفاده از GraphQLLayer
    """
    if analytics_service is None:
        from ..service import create_analytics_service
        analytics_service = create_analytics_service()

    return GraphQLLayer(analytics_service=analytics_service)


def create_rest_api(analytics_service: AnalyticsService = None):
    """
    ایجاد یک نمونه از RestAPI با تنظیمات مناسب

    Args:
        analytics_service (AnalyticsService, optional): سرویس تحلیل داده‌ها

    Returns:
        RestAPI: نمونه آماده استفاده از RestAPI
    """
    if analytics_service is None:
        from ..service import create_analytics_service
        analytics_service = create_analytics_service()

    return RestAPI(analytics_service=analytics_service)


def create_stream_processor(clickhouse_adapter: ClickHouseAdapter = None):
    """
    ایجاد یک نمونه از StreamProcessor با تنظیمات مناسب

    Args:
        clickhouse_adapter (ClickHouseAdapter, optional): آداپتور ClickHouse

    Returns:
        StreamProcessor: نمونه آماده استفاده از StreamProcessor
    """
    if clickhouse_adapter is None:
        from ..adapters import create_adapter
        clickhouse_adapter = create_adapter()

    return StreamProcessor(clickhouse_adapter=clickhouse_adapter)
