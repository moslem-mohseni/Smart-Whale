"""
ماژول زیرساخت (Infrastructure)
این ماژول تمامی سرویس‌های زیرساختی پروژه را مدیریت می‌کند.
"""

from typing import TYPE_CHECKING, Any

def get_kafka_service(*args: Any, **kwargs: Any):
    """به صورت lazy سرویس Kafka را بارگذاری می‌کند"""
    from .kafka import KafkaService
    return KafkaService(*args, **kwargs)

def get_redis_service(*args: Any, **kwargs: Any):
    """به صورت lazy سرویس Redis را بارگذاری می‌کند"""
    from .redis import CacheService
    return CacheService(*args, **kwargs)

def get_timescaledb_service(*args: Any, **kwargs: Any):
    """به صورت lazy سرویس TimescaleDB را بارگذاری می‌کند"""
    from .timescaledb import TimescaleDBService
    return TimescaleDBService(*args, **kwargs)

# تنها چیزهایی که واقعاً نیاز داریم را export می‌کنیم
__all__ = [
    'get_kafka_service',
    'get_redis_service',
    'get_timescaledb_service'
]