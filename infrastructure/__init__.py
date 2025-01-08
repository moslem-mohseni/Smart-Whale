# infrastructure/__init__.py

"""
ماژول زیرساخت (Infrastructure)

این ماژول تمامی سرویس‌های زیرساختی پروژه را مدیریت می‌کند. هر سرویس در یک زیرماژول مجزا
پیاده‌سازی شده و از طریق اینترفیس‌های استاندارد در دسترس قرار می‌گیرد.

سرویس‌های اصلی:
- Kafka: برای مدیریت پیام‌رسانی و ارتباطات ناهمگام
- Redis: برای مدیریت کش و ذخیره‌سازی موقت
- TimescaleDB: برای ذخیره‌سازی داده‌های سری زمانی
- ClickHouse: برای تحلیل‌های پیچیده و پردازش حجم بالای داده

تمام سرویس‌ها از الگوی طراحی تمیز پیروی می‌کنند و از طریق اینترفیس‌های تعریف شده
در ماژول interfaces قابل دسترسی هستند.
"""

from .kafka import KafkaService, KafkaConfig
from .redis import CacheService, RedisConfig
from .timescaledb import TimescaleDBService, TimescaleDBConfig
from .clickhouse import AnalyticsService, ClickHouseConfig
from .interfaces import (
    StorageInterface,
    CachingInterface,
    MessagingInterface,
    InfrastructureError,
    ConnectionError,
    OperationError
)

__version__ = '1.0.0'

__all__ = [
    # سرویس‌های اصلی
    'KafkaService',
    'CacheService',
    'TimescaleDBService',
    'AnalyticsService',

    # کلاس‌های تنظیمات
    'KafkaConfig',
    'RedisConfig',
    'TimescaleDBConfig',
    'ClickHouseConfig',

    # اینترفیس‌ها
    'StorageInterface',
    'CachingInterface',
    'MessagingInterface',

    # کلاس‌های خطا
    'InfrastructureError',
    'ConnectionError',
    'OperationError'
]