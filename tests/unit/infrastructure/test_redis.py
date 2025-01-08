# tests/unit/infrastructure/test_redis.py

import pytest
import pytest_asyncio
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch, MagicMock
from infrastructure.redis import CacheService, RedisConfig, CacheItem, CacheNamespace
from infrastructure.redis.adapters.redis_adapter import RedisAdapter
from infrastructure.interfaces import ConnectionError, OperationError


@pytest.fixture
def redis_config():
    return RedisConfig(
        host="localhost",
        port=6379,
        database=0,
        max_connections=5,
        cluster_mode=False
    )


@pytest.fixture
def cluster_config():
    return RedisConfig(
        host="localhost",
        port=6379,
        database=0,
        cluster_mode=True,
        cluster_nodes=[
            {"host": "localhost", "port": 6379},
            {"host": "localhost", "port": 6380}
        ]
    )


@pytest_asyncio.fixture
async def redis_service(redis_config):
    """فیکسچر برای ایجاد سرویس Redis با mock adapter"""
    service = CacheService(redis_config)
    mock_adapter = AsyncMock()
    mock_adapter.exists = AsyncMock()
    mock_adapter.scan_keys = AsyncMock()
    mock_adapter.ttl = AsyncMock()
    mock_adapter.pipeline = AsyncMock()
    service._adapter = mock_adapter
    yield service
    await service.disconnect()


@pytest.mark.asyncio
async def test_connect_with_retry():
    """تست مکانیزم retry در اتصال"""
    config = RedisConfig(host="localhost", port=6379)

    # ایجاد یک mock adapter به جای استفاده مستقیم از RedisAdapter
    mock_adapter = AsyncMock(spec=RedisAdapter)
    with patch('infrastructure.redis.service.cache_service.RedisAdapter', return_value=mock_adapter):
        service = CacheService(config)
        await service.connect()
        mock_adapter.connect.assert_called_once()


@pytest.mark.asyncio
async def test_cluster_mode_operations(cluster_config):
    """تست عملیات در حالت کلاستر"""
    # ایجاد یک mock adapter به جای استفاده مستقیم از RedisAdapter
    mock_adapter = AsyncMock(spec=RedisAdapter)
    with patch('infrastructure.redis.service.cache_service.RedisAdapter', return_value=mock_adapter):
        service = CacheService(cluster_config)
        await service.connect()
        mock_adapter.connect.assert_called_once()

        # تست عملیات در کلاستر
        await service.set("test_key", "test_value")
        mock_adapter.set.assert_called_once()


@pytest.mark.asyncio
async def test_cleanup_mechanism(redis_service):
    """تست مکانیزم cleanup"""
    # تنظیم یک namespace برای تست
    test_namespace = CacheNamespace("test_namespace", default_ttl=300)
    redis_service.create_namespace(test_namespace)

    # تنظیم mock برای scan_keys
    expired_keys = ["test_namespace:expired1", "test_namespace:expired2"]
    redis_service._adapter.scan_keys.return_value = expired_keys
    redis_service._adapter.ttl.return_value = -1

    # اجرای cleanup
    await redis_service._cleanup_expired_keys()

    # بررسی حذف کلیدهای منقضی شده
    delete_calls = redis_service._adapter.delete.call_args_list
    assert len(delete_calls) == 2
    assert delete_calls[0][0][0] == expired_keys[0]
    assert delete_calls[1][0][0] == expired_keys[1]


@pytest.mark.asyncio
async def test_batch_operations(redis_service):
    """تست عملیات دسته‌ای"""
    batch_data = {
        "key1": "value1",
        "key2": "value2",
        "key3": "value3"
    }

    pipeline_mock = AsyncMock()
    redis_service._adapter.pipeline.return_value = pipeline_mock

    # ذخیره‌سازی دسته‌ای
    await asyncio.gather(*[
        redis_service.set(key, value)
        for key, value in batch_data.items()
    ])

    assert redis_service._adapter.set.call_count == len(batch_data)

    # تست بازیابی دسته‌ای
    redis_service._adapter.get.side_effect = lambda k: batch_data.get(k)

    results = await asyncio.gather(*[
        redis_service.get(key)
        for key in batch_data.keys()
    ])

    assert all(val in batch_data.values() for val in results)


@pytest.mark.asyncio
async def test_namespace_isolation(redis_service):
    """تست ایزوله بودن فضاهای نام"""
    ns1 = CacheNamespace("namespace1", default_ttl=300)
    ns2 = CacheNamespace("namespace2", default_ttl=600)

    redis_service.create_namespace(ns1)
    redis_service.create_namespace(ns2)

    # ذخیره داده در دو فضای نام
    await redis_service.set("key", "value1", namespace="namespace1")
    await redis_service.set("key", "value2", namespace="namespace2")

    # بررسی جداسازی داده‌ها
    redis_service._adapter.get.side_effect = lambda k: {
        "namespace1:key": "value1",
        "namespace2:key": "value2"
    }.get(k)

    value1 = await redis_service.get("key", namespace="namespace1")
    value2 = await redis_service.get("key", namespace="namespace2")

    assert value1 != value2
    assert value1 == "value1"
    assert value2 == "value2"


@pytest.mark.asyncio
async def test_error_recovery(redis_service):
    """تست بازیابی از خطا"""
    redis_service._adapter.set.side_effect = [
        OperationError("Failed"),
        None  # موفق در تلاش دوم
    ]

    # تلاش اول با شکست مواجه می‌شود
    with pytest.raises(OperationError):
        await redis_service.set("key", "value")

    # تلاش دوم موفق است
    redis_service._adapter.set.side_effect = None
    await redis_service.set("key", "value")

    # بررسی وضعیت اتصال
    redis_service._adapter.is_connected.return_value = True
    assert await redis_service._adapter.is_connected()