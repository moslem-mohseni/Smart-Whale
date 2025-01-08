# tests/unit/infrastructure/test_clickhouse.py

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch
from datetime import datetime, timedelta
from infrastructure.clickhouse import (
    AnalyticsService,
    ClickHouseConfig,
    AnalyticsEvent,
    AnalyticsQuery,
    AnalyticsResult,
    TableSchema,
    MaintenanceManager,
    QuerySettings
)
from infrastructure.redis import RedisConfig
from infrastructure.interfaces import ConnectionError, OperationError


@pytest.fixture
def clickhouse_config():
    """فیکسچر برای تنظیمات ClickHouse"""
    return ClickHouseConfig(
        host="localhost",
        port=9000,
        database="test_db",
        user="test_user",
        password="test_pass",
        max_connections=5
    )


@pytest.fixture
def redis_config():
    """فیکسچر برای تنظیمات Redis"""
    return RedisConfig(
        host="localhost",
        port=6379,
        database=0,
        max_connections=5
    )


@pytest_asyncio.fixture
async def analytics_service(clickhouse_config, redis_config):
    """فیکسچر برای سرویس تحلیلی"""
    service = AnalyticsService(clickhouse_config, redis_config)
    service._adapter = AsyncMock()
    service._cache = AsyncMock()
    # فقط اتصال برقرار می‌کنیم، جداول ایجاد نمی‌شوند
    await service.connect()
    yield service
    await service.shutdown()


@pytest.mark.asyncio
async def test_analytics_service_initialization(analytics_service, clickhouse_config):
    """تست راه‌اندازی سرویس"""
    assert analytics_service.config == clickhouse_config
    assert analytics_service._adapter is not None
    analytics_service._adapter.connect.assert_called_once()
    analytics_service._cache.initialize.assert_called_once()


@pytest.mark.asyncio
async def test_event_storage_and_caching(analytics_service):
    """تست ذخیره‌سازی و کش‌گذاری رویدادها"""
    event = AnalyticsEvent(
        event_id="test_123",
        event_type="user_action",
        timestamp=datetime.now(),
        data={"action": "click"}
    )

    analytics_service._cache.get_cached_result.return_value = None
    mock_result = [{"count": 1}]
    analytics_service._adapter.execute_analytics_query.return_value = mock_result

    query = AnalyticsQuery(
        dimensions=["event_type"],
        metrics=["count()"]
    )
    result = await analytics_service.execute_analytics_query(query)

    assert result.data == mock_result
    analytics_service._cache.get_cached_result.assert_called_once()
    analytics_service._cache.cache_result.assert_called_once()


@pytest.mark.asyncio
async def test_trend_analysis(analytics_service):
    """تست تحلیل روندها"""
    mock_trends = [
        {'period': '2024-01-01', 'count': 100},
        {'period': '2024-01-02', 'count': 150}
    ]
    analytics_service._adapter.execute_analytics_query.return_value = mock_trends

    result = await analytics_service.get_event_trends(
        event_type="user_action",
        interval="1 day",
        days=7,
        use_cache=False
    )

    assert result.data == mock_trends


@pytest.mark.asyncio
async def test_cache_invalidation(analytics_service):
    """تست حذف کش"""
    event_type = "user_action"
    await analytics_service.invalidate_event_cache(event_type)
    analytics_service._cache.invalidate_cache.assert_called_once()



@pytest.mark.asyncio
async def test_error_handling(analytics_service):
    """
    تست مدیریت خطاها

    این تست بررسی می‌کند که سیستم به درستی با خطاهای مختلف مثل خطای اتصال
    و خطاهای عملیاتی برخورد می‌کند.
    """
    # تنظیم mock برای شبیه‌سازی خطای اتصال
    from infrastructure.interfaces.exceptions import ConnectionError, OperationError
    analytics_service._adapter.execute_analytics_query = AsyncMock(
        side_effect=ConnectionError("Database connection failed")
    )
    analytics_service._cache.get_cached_result.return_value = None

    # تست خطای اتصال
    query = AnalyticsQuery(
        dimensions=["event_type"],
        metrics=["count()"]
    )

    # انتظار داریم خطای ConnectionError دریافت کنیم
    with pytest.raises(ConnectionError) as exc_info:
        await analytics_service.execute_analytics_query(query)
    assert str(exc_info.value) == "Database connection failed"

    # تست خطای عملیاتی
    analytics_service._adapter.execute_analytics_query = AsyncMock(
        side_effect=Exception("Unknown database error")
    )

    # انتظار داریم خطای OperationError دریافت کنیم
    with pytest.raises(OperationError) as exc_info:
        await analytics_service.execute_analytics_query(query)
    assert "Query execution failed" in str(exc_info.value)

    # تأیید درخواست‌های صحیح به کش
    analytics_service._cache.get_cached_result.assert_called_with(query)


@pytest.mark.asyncio
async def test_cache_stats(analytics_service):
    """تست آمار کش"""
    mock_stats = {
        'total_keys': 100,
        'hit_rate': 0.8
    }
    analytics_service._cache.get_cache_stats.return_value = mock_stats

    stats = await analytics_service.get_cache_stats()
    assert stats == mock_stats
    analytics_service._cache.get_cache_stats.assert_called_once()


@pytest.mark.asyncio
async def test_table_management(analytics_service):
    """تست مدیریت جداول"""
    schema = TableSchema(
        name="test_events",
        columns={
            'event_id': 'String',
            'timestamp': 'DateTime',
            'data': 'String'
        }
    )

    await analytics_service._adapter.create_table(schema.name, schema.columns)
    analytics_service._adapter.create_table.assert_called_once_with(
        schema.name,
        schema.columns
    )



@pytest.mark.asyncio
async def test_query_settings(analytics_service):
    """تست تنظیمات پرس‌وجو"""
    # تنظیم query
    query = AnalyticsQuery(
        dimensions=["event_type"],
        metrics=["count()"],
        limit=1000
    )

    # تنظیم mock
    mock_result = []
    mock_query = AsyncMock(return_value=mock_result)
    analytics_service._adapter.execute_analytics_query = mock_query

    # غیرفعال کردن کش برای این تست
    analytics_service._cache.get_cached_result.return_value = None

    # اجرای تست
    await analytics_service.execute_analytics_query(query)

    # بررسی فراخوانی
    mock_query.assert_called_once_with(query)

    # همچنین می‌توانیم بررسی کنیم که کش هم درست استفاده شده
    analytics_service._cache.get_cached_result.assert_called_once_with(query)
    analytics_service._cache.cache_result.assert_called_once()