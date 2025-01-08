# tests/unit/infrastructure/test_timescaledb.py

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch
from datetime import datetime, timedelta
from infrastructure.timescaledb import (
    TimescaleDBService,
    TimescaleDBConfig,
    TimeSeriesData,
    TimeSeriesRepository,
    TableSchema
)
from infrastructure.interfaces import ConnectionError, OperationError


@pytest.fixture
def timescale_config():
    """فیکسچر برای ایجاد تنظیمات TimescaleDB"""
    return TimescaleDBConfig(
        host="localhost",
        port=5432,
        database="test_timescale",
        user="test_user",
        password="test_pass",
        min_connections=1,
        max_connections=5
    )


@pytest_asyncio.fixture
async def timescale_service(timescale_config):
    """فیکسچر برای ایجاد سرویس TimescaleDB با mock adapter"""
    service = TimescaleDBService(timescale_config)
    mock_storage = AsyncMock()
    service._storage = mock_storage
    await service.initialize()
    yield service
    await service.shutdown()


@pytest.mark.asyncio
async def test_service_initialization(timescale_service, timescale_config):
    """تست راه‌اندازی اولیه سرویس"""
    assert timescale_service.config == timescale_config
    assert timescale_service._storage is not None
    timescale_service._storage.connect.assert_called_once()


@pytest.mark.asyncio
async def test_time_series_data_storage(timescale_service):
    """تست ذخیره‌سازی داده‌های سری زمانی"""
    test_data = TimeSeriesData(
        id="test_123",
        timestamp=datetime.now(),
        value=42.5,
        metadata={"sensor": "temperature"}
    )

    timescale_service._storage.execute.return_value = [{"id": test_data.id}]
    await timescale_service.store_time_series_data(test_data)
    assert timescale_service._storage.execute.called


@pytest.mark.asyncio
async def test_time_bucket_aggregation(timescale_service):
    """تست تجمیع داده‌ها با استفاده از time bucket"""
    interval = '1 hour'
    start_time = datetime.now() - timedelta(days=1)
    end_time = datetime.now()

    mock_results = [
        {'bucket': '2024-01-05 00:00:00', 'avg_value': 35.5},
        {'bucket': '2024-01-05 01:00:00', 'avg_value': 36.2}
    ]
    timescale_service._storage.execute.return_value = mock_results

    result = await timescale_service.aggregate_time_series(
        metric='value',
        interval=interval,
        start_time=start_time,
        end_time=end_time
    )
    assert len(result) == len(mock_results)


@pytest.mark.asyncio
async def test_continuous_aggregates(timescale_service):
    """تست مدیریت تجمیع‌های مستمر"""
    await timescale_service.create_continuous_aggregate(
        view_name="hourly_metrics",
        table_name="time_series_data",
        interval="1 hour",
        aggregates=["avg(value)", "max(value)", "min(value)"]
    )
    assert timescale_service._storage.execute.called


@pytest.mark.asyncio
async def test_data_retention_policy(timescale_service):
    """تست مدیریت سیاست‌های نگهداری داده"""
    retention_days = 30
    await timescale_service.set_retention_policy(
        table_name="time_series_data",
        interval=f"{retention_days} days"
    )
    assert timescale_service._storage.execute.called


@pytest.mark.asyncio
async def test_compression_policy(timescale_service):
    """تست مدیریت سیاست‌های فشرده‌سازی"""
    await timescale_service.set_compression_policy(
        table_name="time_series_data",
        segment_by="id",
        order_by="timestamp DESC"
    )
    assert timescale_service._storage.execute.called


@pytest.mark.asyncio
async def test_error_handling(timescale_service):
    """تست مدیریت خطاها"""
    timescale_service._storage.execute.side_effect = OperationError("Query failed")

    with pytest.raises(OperationError):
        await timescale_service.store_time_series_data(
            TimeSeriesData(
                id="test",
                timestamp=datetime.now(),
                value=1.0
            )
        )


@pytest.mark.asyncio
async def test_hypertable_management(timescale_service):
    """تست مدیریت hypertable‌ها"""
    schema = TableSchema(
        name="sensor_data",
        columns={
            'time': 'TIMESTAMPTZ',
            'sensor_id': 'TEXT',
            'value': 'DOUBLE PRECISION'
        },
        time_column="time"
    )

    await timescale_service.create_hypertable(
        schema=schema,
        partition_interval="1 day"
    )

    assert timescale_service._storage.create_table.called
    assert timescale_service._storage.create_hypertable.called