"""
ماژول `infrastructure/` مسئول مدیریت ارتباطات و پردازش‌های زیرساختی مرتبط با پردازش زبان است.

📌 اجزای اصلی این ماژول:
- `caching/` → مدیریت کش داده‌های پردازشی
- `vector_store/` → پردازش برداری برای داده‌های زبانی
- `messaging/` → مدیریت ارتباط Kafka برای پردازش داده‌های زبانی
- `timescaledb/` → مدیریت ذخیره‌سازی داده‌های سری‌زمانی
- `file_management/` → مدیریت ذخیره‌سازی فایل‌های پردازشی زبان
- `monitoring/` → مانیتورینگ و بررسی سلامت سرویس‌ها
- `clickhouse/` → ارتباط با پایگاه داده تحلیلی ClickHouse
"""

from .caching import cache_manager, redis_adapter
from .vector_store import vector_search, milvus_adapter
from .messaging import kafka_producer, kafka_consumer
from .timescaledb import timescaledb_adapter, metrics_handler
from .file_management import file_management_service, file_store
from .monitoring import health_check, performance_metrics
from .clickhouse import clickhouse_db

__all__ = [
    # Caching
    "cache_manager",
    "redis_adapter",

    # Vector Store
    "vector_search",
    "milvus_adapter",

    # Messaging
    "kafka_producer",
    "kafka_consumer",

    # TimescaleDB
    "timescaledb_adapter",
    "metrics_handler",

    # File Management
    "file_management_service",
    "file_store",

    # Monitoring
    "health_check",
    "performance_metrics",

    # ClickHouse
    "clickhouse_db",
]
