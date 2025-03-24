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
- Vector Store: برای مدیریت داده‌های برداری و جستجوی معنایی
- File Management: برای مدیریت فایل‌ها و ذخیره‌سازی اسناد

تمام سرویس‌ها از الگوی طراحی تمیز پیروی می‌کنند و از طریق اینترفیس‌های تعریف شده
در ماژول interfaces قابل دسترسی هستند.
"""

import logging

logger = logging.getLogger(__name__)
logger.info("Initializing Infrastructure Module...")

# واردسازی سرویس‌های اصلی
try:
    from .kafka import KafkaService, KafkaConfig
except ImportError as e:
    logger.warning(f"Could not import Kafka module: {e}")
    KafkaService = None
    KafkaConfig = None

try:
    from .redis import CacheService, RedisConfig
except ImportError as e:
    logger.warning(f"Could not import Redis module: {e}")
    CacheService = None
    RedisConfig = None

try:
    from .timescaledb import TimescaleDBService, TimescaleDBConfig
except ImportError as e:
    logger.warning(f"Could not import TimescaleDB module: {e}")
    TimescaleDBService = None
    TimescaleDBConfig = None

try:
    from .clickhouse import AnalyticsService, ClickHouseConfig
except ImportError as e:
    logger.warning(f"Could not import ClickHouse module: {e}")
    AnalyticsService = None
    ClickHouseConfig = None

# واردسازی سرویس‌های Vector Store
try:
    from .vector_store.service.vector_service import VectorService
    from .vector_store.service.index_service import IndexService
    from .vector_store.service.search_service import SearchService
    from .vector_store.backup import BackupService, RestoreService, BackupScheduler
    from .vector_store.monitoring import Metrics as VectorMetrics
    from .vector_store.monitoring import HealthCheck as VectorHealthCheck
    from .vector_store.monitoring import PerformanceLogger
    from .vector_store.optimization import CacheManager
    from .vector_store.migrations import MigrationManager, MigrationVersions
    from .vector_store.config import VectorStoreConfig
except ImportError as e:
    logger.warning(f"Could not import Vector Store module: {e}")
    VectorService, IndexService, SearchService = None, None, None
    BackupService, RestoreService, BackupScheduler = None, None, None
    VectorMetrics, VectorHealthCheck, PerformanceLogger = None, None, None
    CacheManager, MigrationManager, MigrationVersions = None, None, None
    VectorStoreConfig = None

# واردسازی سرویس‌های مدیریت فایل
try:
    from .file_management.service import FileService, StorageService, MetadataService
    from .file_management.config import FileManagerConfig
except ImportError as e:
    logger.warning(f"Could not import File Management module: {e}")
    FileService, StorageService, MetadataService = None, None, None
    FileManagerConfig = None

# واردسازی اینترفیس‌ها
try:
    from .interfaces import (
        StorageInterface,
        CachingInterface,
        MessagingInterface,
        VectorInterface,
        FileInterface,
        InfrastructureError,
        ConnectionError,
        OperationError
    )
except ImportError as e:
    logger.warning(f"Could not import interfaces: {e}")
    StorageInterface, CachingInterface, MessagingInterface = None, None, None
    VectorInterface, FileInterface = None, None
    InfrastructureError, ConnectionError, OperationError = None, None, None

__version__ = '1.0.0'

__all__ = [
    # سرویس‌های اصلی
    'KafkaService',
    'CacheService',
    'TimescaleDBService',
    'AnalyticsService',

    # سرویس‌های Vector Store
    'VectorService',
    'IndexService',
    'SearchService',
    'BackupService',
    'RestoreService',
    'BackupScheduler',
    'VectorMetrics',
    'VectorHealthCheck',
    'PerformanceLogger',
    'CacheManager',
    'MigrationManager',
    'MigrationVersions',

    # سرویس‌های مدیریت فایل
    'FileService',
    'StorageService',
    'MetadataService',

    # کلاس‌های تنظیمات
    'KafkaConfig',
    'RedisConfig',
    'TimescaleDBConfig',
    'ClickHouseConfig',
    'VectorStoreConfig',
    'FileManagerConfig',

    # اینترفیس‌ها
    'StorageInterface',
    'CachingInterface',
    'MessagingInterface',
    'VectorInterface',
    'FileInterface',

    # کلاس‌های خطا
    'InfrastructureError',
    'ConnectionError',
    'OperationError'
]
