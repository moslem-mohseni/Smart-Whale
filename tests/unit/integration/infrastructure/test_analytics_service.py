# tests/integration/infrastructure/test_analytics_service.py

import pytest
import pytest_asyncio
import asyncio
from datetime import datetime, timedelta
from typing import AsyncGenerator, List, Dict, Any
from functools import wraps

from infrastructure.clickhouse import (
    AnalyticsService, ClickHouseConfig, AnalyticsEvent, AnalyticsQuery
)
from infrastructure.redis import RedisConfig
from .test_env import TestEnvironmentManager


def async_test(f):
    """دکوراتور برای مدیریت event loop در تست‌های async"""

    @wraps(f)
    def wrapper(*args, **kwargs):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(f(*args, **kwargs))
        finally:
            loop.close()

    return wrapper


@pytest.fixture(scope="session")
def event_loop():
    """فیکسچر event loop در سطح session"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def env_manager(event_loop) -> AsyncGenerator[TestEnvironmentManager, None]:
    """فیکسچر مدیریت محیط تست"""
    manager = TestEnvironmentManager()
    await manager.setup()
    yield manager
    await manager.cleanup()


@pytest_asyncio.fixture(scope="session")
async def service_config(env_manager) -> Dict[str, Any]:
    """فیکسچر تنظیمات سرویس"""
    return {
        'clickhouse': ClickHouseConfig(
            host="localhost",
            port=9000,
            database="test_db",
            user="test_user",
            password="test_pass"
        ),
        'redis': RedisConfig(
            host="localhost",
            port=6379,
            database=0
        )
    }


@pytest_asyncio.fixture(scope="function")
async def analytics_service(service_config) -> AsyncGenerator[AnalyticsService, None]:
    """فیکسچر سرویس تحلیلی برای هر تست"""
    service = AnalyticsService(
        service_config['clickhouse'],
        service_config['redis']
    )
    await service.initialize()
    try:
        yield service
    finally:
        await service.shutdown()


async def generate_test_events(count: int = 100) -> List[AnalyticsEvent]:
    """تولید داده‌های تست"""
    events = []
    event_types = ['page_view', 'click', 'scroll', 'form_submit']
    base_time = datetime.now()

    for i in range(count):
        event_type = event_types[i % len(event_types)]
        event_time = base_time + timedelta(minutes=i, seconds=i * 2)

        data = {
            'user_id': f"user_{i % 10}",
            'session_id': f"session_{i % 5}",
            'page': f"/page_{i % 20}",
            'duration': i * 10,
            'device_type': ['mobile', 'desktop', 'tablet'][i % 3]
        }

        events.append(AnalyticsEvent(
            event_id=f"evt_{i}",
            event_type=event_type,
            timestamp=event_time,
            data=data,
            metadata={'version': '1.0', 'environment': 'test'}
        ))

    return events


@pytest.mark.asyncio
@pytest.mark.integration
async def test_event_storage_and_retrieval(analytics_service):
    """تست ذخیره‌سازی و بازیابی رویدادها"""
    test_events = await generate_test_events(100)
    start_time = datetime.now()

    for event in test_events:
        await analytics_service.store_event(event)

    storage_time = (datetime.now() - start_time).total_seconds()
    assert storage_time < 5

    event = test_events[0]
    cached_result = await analytics_service.get_event(event.event_id)
    assert cached_result is not None
    assert cached_result['event_id'] == event.event_id

    await analytics_service.invalidate_event_cache(event.event_type)
    db_result = await analytics_service.get_event(event.event_id)
    assert db_result is not None
    assert db_result['event_id'] == event.event_id


@pytest.mark.asyncio
@pytest.mark.integration
async def test_analytics_query_performance(analytics_service):
    """تست عملکرد کوئری‌های تحلیلی"""
    test_events = await generate_test_events(1000)
    for event in test_events:
        await analytics_service.store_event(event)

    query = AnalyticsQuery(
        dimensions=['event_type', 'data.device_type'],
        metrics=[
            'count() as event_count',
            'avg(data.duration) as avg_duration',
            'uniq(data.user_id) as unique_users'
        ],
        filters={
            'timestamp': {
                'start': datetime.now() - timedelta(hours=1),
                'end': datetime.now()
            },
            'data.duration': {'min': 50}
        },
        order_by=['event_count DESC'],
        limit=100
    )

    start_time = datetime.now()
    first_result = await analytics_service.execute_analytics_query(query)
    first_execution_time = (datetime.now() - start_time).total_seconds()

    start_time = datetime.now()
    cached_result = await analytics_service.execute_analytics_query(query)
    cached_execution_time = (datetime.now() - start_time).total_seconds()

    assert cached_execution_time < first_execution_time * 0.5
    assert first_execution_time < 5
    assert cached_execution_time < 0.5


@pytest.mark.asyncio
@pytest.mark.integration
async def test_concurrent_operations(analytics_service):
    """تست عملیات همزمان"""
    events = await generate_test_events(500)
    queries = [
        AnalyticsQuery(
            dimensions=['event_type'],
            metrics=['count()'],
            filters={'event_type': event.event_type}
        )
        for event in events[:10]
    ]

    async def mixed_operations():
        await asyncio.gather(
            *[analytics_service.store_event(event) for event in events],
            *[analytics_service.execute_analytics_query(query) for query in queries]
        )

    start_time = datetime.now()
    await asyncio.gather(*[mixed_operations() for _ in range(3)])
    total_time = (datetime.now() - start_time).total_seconds()

    operations_count = len(events) * 3 + len(queries) * 3
    operations_per_second = operations_count / total_time
    assert operations_per_second > 100


@pytest.mark.asyncio
@pytest.mark.integration
async def test_error_recovery_and_data_consistency(analytics_service, env_manager):
    """تست بازیابی از خطا و سازگاری داده‌ها"""
    initial_events = await generate_test_events(10)
    for event in initial_events:
        await analytics_service.store_event(event)

    await env_manager.stop_service('redis')
    await asyncio.sleep(2)
    await env_manager.start_service('redis')
    await asyncio.sleep(2)

    new_events = await generate_test_events(10)
    for event in new_events:
        await analytics_service.store_event(event)

    for event in initial_events + new_events:
        result = await analytics_service.get_event(event.event_id)
        assert result is not None
        assert result['event_id'] == event.event_id