# tests/integration/infrastructure/test_cache_service.py

import pytest
import asyncio
import docker
from typing import AsyncGenerator
from infrastructure.redis import CacheService, RedisConfig, CacheNamespace


class DockerServiceManager:
    """
    مدیریت سرویس‌های داکر برای تست‌های یکپارچگی

    این کلاس مسئول راه‌اندازی و مدیریت کانتینرهای مورد نیاز برای تست است.
    از Docker SDK برای مدیریت کانتینرها استفاده می‌کند.
    """

    def __init__(self):
        self.client = docker.from_env()
        self.containers = {}

    async def start_redis(self) -> dict:
        """راه‌اندازی کانتینر Redis برای تست"""
        container = self.client.containers.run(
            'redis:6.2-alpine',
            ports={'6379/tcp': 6379},
            detach=True,
            remove=True,
            name='test_redis'
        )
        self.containers['redis'] = container
        # انتظار برای آماده شدن Redis
        await asyncio.sleep(2)
        return {'host': 'localhost', 'port': 6379}

    async def cleanup(self):
        """پاکسازی و حذف تمام کانتینرها"""
        for container in self.containers.values():
            container.stop()
        self.containers.clear()


@pytest.fixture(scope="session")
async def docker_services() -> AsyncGenerator[DockerServiceManager, None]:
    """فیکسچر برای مدیریت سرویس‌های داکر"""
    manager = DockerServiceManager()
    yield manager
    await manager.cleanup()


@pytest.fixture(scope="session")
async def redis_config(docker_services) -> RedisConfig:
    """فیکسچر برای تنظیمات Redis"""
    redis_params = await docker_services.start_redis()
    return RedisConfig(
        host=redis_params['host'],
        port=redis_params['port'],
        database=0,
        max_connections=5
    )


@pytest.fixture
async def cache_service(redis_config) -> AsyncGenerator[CacheService, None]:
    """فیکسچر برای سرویس کش"""
    service = CacheService(redis_config)
    await service.initialize()
    yield service
    await service.disconnect()


@pytest.mark.integration
async def test_cache_basic_operations(cache_service):
    """
    تست عملیات پایه کش

    این تست عملیات اصلی مانند ذخیره‌سازی، بازیابی و حذف داده را
    با استفاده از یک نمونه واقعی Redis بررسی می‌کند.
    """
    test_key = "test_key"
    test_value = {"name": "test", "value": 42}

    # تست ذخیره‌سازی و بازیابی
    await cache_service.set(test_key, test_value)
    result = await cache_service.get(test_key)
    assert result == test_value

    # تست حذف
    await cache_service.delete(test_key)
    result = await cache_service.get(test_key)
    assert result is None


@pytest.mark.integration
async def test_cache_with_ttl(cache_service):
    """
    تست مدیریت زمان انقضا

    بررسی می‌کند که مکانیزم TTL (Time To Live) به درستی کار می‌کند
    و داده‌ها پس از گذشت زمان مشخص شده منقضی می‌شوند.
    """
    test_key = "ttl_test"
    test_value = "will expire"
    ttl = 2  # ثانیه

    await cache_service.set(test_key, test_value, ttl=ttl)

    # بررسی وجود داده قبل از انقضا
    result = await cache_service.get(test_key)
    assert result == test_value

    # انتظار برای انقضای داده
    await asyncio.sleep(ttl + 1)

    # بررسی عدم وجود داده پس از انقضا
    result = await cache_service.get(test_key)
    assert result is None


@pytest.mark.integration
async def test_cache_namespaces(cache_service):
    """
    تست مدیریت فضای‌نام

    بررسی می‌کند که سیستم می‌تواند داده‌ها را در فضاهای نام مختلف
    ذخیره و مدیریت کند و تنظیمات هر فضای نام به درستی اعمال می‌شود.
    """
    namespace = CacheNamespace(
        name="test_space",
        default_ttl=300,
        max_size=1000
    )

    cache_service.create_namespace(namespace)

    # تست ذخیره‌سازی در فضای نام
    test_key = "ns_test"
    test_value = "namespace value"

    await cache_service.set(test_key, test_value, namespace=namespace.name)
    result = await cache_service.get(test_key, namespace=namespace.name)
    assert result == test_value


@pytest.mark.integration
async def test_cache_concurrent_access(cache_service):
    """
    تست دسترسی همزمان

    بررسی می‌کند که سیستم کش می‌تواند به درستی درخواست‌های همزمان
    را مدیریت کند و از race condition جلوگیری کند.
    """
    test_key = "concurrent_test"
    iterations = 100

    async def increment():
        current = await cache_service.get(test_key) or 0
        await cache_service.set(test_key, current + 1)

    # اجرای همزمان چندین عملیات افزایش
    tasks = [increment() for _ in range(iterations)]
    await asyncio.gather(*tasks)

    # بررسی نتیجه نهایی
    final_value = await cache_service.get(test_key)
    assert final_value == iterations


@pytest.mark.integration
async def test_cache_large_data(cache_service):
    """
    تست داده‌های حجیم

    بررسی می‌کند که سیستم می‌تواند داده‌های نسبتاً بزرگ را
    به درستی ذخیره و بازیابی کند.
    """
    test_key = "large_data"
    # ایجاد یک دیکشنری بزرگ برای تست
    large_data = {f"key_{i}": "x" * 1000 for i in range(1000)}

    await cache_service.set(test_key, large_data)
    result = await cache_service.get(test_key)
    assert result == large_data