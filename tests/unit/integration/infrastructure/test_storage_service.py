# tests/integration/infrastructure/test_storage_service.py

import pytest
import asyncio
import docker
import numpy as np
from datetime import datetime, timedelta
from typing import AsyncGenerator, List, Dict
from infrastructure.timescaledb import (
    TimescaleDBService,
    TimescaleDBConfig,
    TimeSeriesData,
    TableSchema
)


class TimescaleDBTestEnvironment:
    """
    مدیریت محیط تست TimescaleDB

    این کلاس یک نمونه TimescaleDB را در داکر راه‌اندازی می‌کند و امکانات لازم برای
    تست داده‌های سری زمانی را فراهم می‌کند. این محیط ویژگی‌های خاص TimescaleDB مانند
    hypertable و تجمیع مستمر را پشتیبانی می‌کند.
    """

    def __init__(self):
        self.client = docker.from_env()
        self.container = None

    async def setup(self) -> dict:
        """راه‌اندازی محیط تست"""
        self.container = self.client.containers.run(
            'timescale/timescaledb:latest-pg13',
            ports={'5432/tcp': 5432},
            environment={
                'POSTGRES_DB': 'test_db',
                'POSTGRES_USER': 'test_user',
                'POSTGRES_PASSWORD': 'test_password'
            },
            detach=True,
            remove=True,
            name='test_timescaledb'
        )

        # انتظار برای آماده شدن دیتابیس
        await asyncio.sleep(10)

        return {
            'host': 'localhost',
            'port': 5432,
            'database': 'test_db',
            'user': 'test_user',
            'password': 'test_password'
        }

    async def cleanup(self):
        """پاکسازی محیط تست"""
        if self.container:
            self.container.stop()


def generate_sensor_data(num_points: int = 1000) -> List[TimeSeriesData]:
    """
    تولید داده‌های شبیه‌سازی شده سنسور

    این تابع داده‌های سری زمانی واقع‌گرایانه تولید می‌کند که شامل چندین متغیر
    و الگوهای زمانی مختلف است.
    """
    base_time = datetime.now() - timedelta(days=1)
    time_points = [base_time + timedelta(minutes=i) for i in range(num_points)]

    # شبیه‌سازی داده‌های سنسور با نویز و روند
    temperature = np.sin(np.linspace(0, 4 * np.pi, num_points)) * 10 + 25 + np.random.normal(0, 0.5, num_points)
    humidity = np.cos(np.linspace(0, 2 * np.pi, num_points)) * 20 + 60 + np.random.normal(0, 1, num_points)

    return [
        TimeSeriesData(
            id=f"sensor_{i}",
            timestamp=time_points[i],
            value={
                'temperature': float(temperature[i]),
                'humidity': float(humidity[i])
            },
            metadata={'location': 'test_room', 'sensor_type': 'environmental'}
        )
        for i in range(num_points)
    ]


@pytest.fixture(scope="session")
async def timescaledb_environment() -> AsyncGenerator[TimescaleDBTestEnvironment, None]:
    """فیکسچر برای مدیریت محیط تست"""
    env = TimescaleDBTestEnvironment()
    await env.setup()
    yield env
    await env.cleanup()


@pytest.fixture(scope="session")
async def storage_config(timescaledb_environment) -> TimescaleDBConfig:
    """فیکسچر برای تنظیمات TimescaleDB"""
    params = await timescaledb_environment.setup()
    return TimescaleDBConfig(**params)


@pytest.fixture
async def storage_service(storage_config) -> AsyncGenerator[TimescaleDBService, None]:
    """فیکسچر برای سرویس ذخیره‌سازی داده‌های سری زمانی"""
    service = TimescaleDBService(storage_config)
    await service.initialize()
    yield service
    await service.shutdown()


@pytest.mark.integration
async def test_hypertable_creation_and_data_insertion(storage_service):
    """
    تست ایجاد hypertable و درج داده

    بررسی می‌کند که سیستم می‌تواند hypertable ایجاد کند و داده‌ها را
    با کارایی بالا در آن درج کند.
    """
    # تعریف schema برای داده‌های سنسور
    schema = TableSchema(
        name="sensor_data",
        columns={
            'time': 'TIMESTAMPTZ',
            'sensor_id': 'TEXT',
            'temperature': 'DOUBLE PRECISION',
            'humidity': 'DOUBLE PRECISION'
        },
        partition_interval='1 day'
    )

    # ایجاد hypertable
    await storage_service.create_hypertable(schema, 'time')

    # درج داده‌های تست
    test_data = generate_sensor_data(100)
    for data in test_data:
        await storage_service.insert_data(
            schema.name,
            {
                'time': data.timestamp,
                'sensor_id': data.id,
                **data.value
            }
        )

    # بررسی درج داده‌ها
    result = await storage_service.execute_query(
        f"SELECT count(*) as count FROM {schema.name}"
    )
    assert result[0]['count'] == 100


@pytest.mark.integration
async def test_time_bucketing_and_aggregation(storage_service):
    """
    تست تجمیع داده‌ها در بازه‌های زمانی

    بررسی می‌کند که سیستم می‌تواند داده‌ها را در بازه‌های زمانی مختلف
    تجمیع کند و نتایج درست برگرداند.
    """
    # تولید و درج داده‌های بیشتر برای تجمیع
    test_data = generate_sensor_data(1000)

    # تجمیع داده‌ها در بازه‌های یک ساعته
    result = await storage_service.aggregate_time_series(
        table_name="sensor_data",
        interval='1 hour',
        metrics=[
            'avg(temperature) as avg_temp',
            'avg(humidity) as avg_humid'
        ],
        group_by=['sensor_id']
    )

    assert len(result) > 0
    assert 'avg_temp' in result[0]
    assert 'avg_humid' in result[0]


@pytest.mark.integration
async def test_continuous_aggregates(storage_service):
    """
    تست تجمیع مستمر

    بررسی می‌کند که سیستم می‌تواند تجمیع‌های مستمر ایجاد کند و
    آن‌ها را به صورت خودکار به‌روز نگه دارد.
    """
    # ایجاد تجمیع مستمر برای میانگین ساعتی دما
    await storage_service.create_continuous_aggregate(
        view_name="hourly_temperature",
        query="""
        SELECT time_bucket('1 hour', time) as bucket,
               avg(temperature) as avg_temp
        FROM sensor_data
        GROUP BY bucket
        """
    )

    # افزودن داده‌های جدید
    new_data = generate_sensor_data(100)
    for data in new_data:
        await storage_service.insert_data(
            "sensor_data",
            {
                'time': data.timestamp,
                'sensor_id': data.id,
                **data.value
            }
        )

    # بررسی به‌روزرسانی تجمیع
    result = await storage_service.execute_query(
        "SELECT count(*) as count FROM hourly_temperature"
    )
    assert result[0]['count'] > 0


@pytest.mark.integration
async def test_data_retention_and_compression(storage_service):
    """
    تست نگهداری و فشرده‌سازی داده‌ها

    بررسی می‌کند که سیستم می‌تواند سیاست‌های نگهداری داده را اعمال کند
    و داده‌های قدیمی را به درستی مدیریت کند.
    """
    # تنظیم سیاست نگهداری
    await storage_service.set_retention_policy(
        table_name="sensor_data",
        interval="7 days"
    )

    # فعال‌سازی فشرده‌سازی
    await storage_service.enable_compression(
        table_name="sensor_data",
        compress_after="2 days"
    )

    # بررسی وضعیت فشرده‌سازی
    chunks_info = await storage_service.get_chunks_info("sensor_data")
    assert chunks_info is not None


@pytest.mark.integration
async def test_performance_under_load(storage_service):
    """
    تست عملکرد تحت بار

    بررسی می‌کند که سیستم می‌تواند حجم بالای داده و درخواست‌های همزمان
    را با کارایی مناسب مدیریت کند.
    """
    # تولید حجم زیادی داده
    bulk_data = generate_sensor_data(10000)

    # درج همزمان داده‌ها
    start_time = datetime.now()
    tasks = [
        storage_service.insert_data(
            "sensor_data",
            {
                'time': data.timestamp,
                'sensor_id': data.id,
                **data.value
            }
        )
        for data in bulk_data
    ]
    await asyncio.gather(*tasks)
    insertion_time = (datetime.now() - start_time).total_seconds()

    # بررسی زمان درج
    assert insertion_time < 60  # حداکثر 60 ثانیه برای درج 10000 رکورد

    # اجرای چند پرس‌وجوی همزمان
    query_tasks = [
        storage_service.aggregate_time_series(
            table_name="sensor_data",
            interval='1 hour',
            metrics=['avg(temperature)']
        ) for _ in range(5)
    ]

    query_results = await asyncio.gather(*query_tasks)
    assert all(len(result) > 0 for result in query_results)