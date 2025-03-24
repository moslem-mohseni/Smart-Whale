# infrastructure/clickhouse/__init__.py
"""
ماژول اصلی ClickHouse

این ماژول تمام قابلیت‌های لازم برای کار با پایگاه داده ClickHouse را فراهم می‌کند:
- اتصال و مدیریت تنظیمات
- اجرای کوئری‌های تحلیلی و مدیریت داده‌ها
- بهینه‌سازی و کش‌گذاری کوئری‌ها
- مانیتورینگ و سنجش عملکرد
- پشتیبان‌گیری و مدیریت چرخه حیات داده‌ها
- یکپارچه‌سازی با GraphQL، REST API و پردازش جریانی
- امنیت و کنترل دسترسی
"""

import logging
from typing import Optional

# زیرماژول‌های اصلی
from .config import ClickHouseConfig
from .config.config import config

# آداپتورها
from .adapters import (
    ClickHouseConnectionPool,
    CircuitBreaker,
    RetryHandler,
    ClickHouseLoadBalancer,
    ClickHouseAdapter,
    create_adapter
)

# دامنه
from .domain import AnalyticsQuery, AnalyticsResult

# یکپارچه‌سازی
from .integration import (
    GraphQLLayer,
    RestAPI,
    StreamProcessor,
    create_graphql_layer,
    create_rest_api,
    create_stream_processor
)

# مدیریت
from .management import (
    BackupManager,
    DataLifecycleManager,
    MigrationManager,
    create_backup_manager,
    create_lifecycle_manager,
    create_migration_manager
)

# مانیتورینگ
from .monitoring import (
    PerformanceMonitor,
    HealthCheck,
    PrometheusExporter,
    start_monitoring
)

# بهینه‌سازی
from .optimization import (
    DataCompressor,
    QueryOptimizer
)

# امنیت
from .security import (
    AccessControl,
    AuditLogger,
    EncryptionManager,
    create_encryption_manager,
    create_access_control,
    create_audit_logger
)

# سرویس‌ها
from .service import (
    AnalyticsCache,
    AnalyticsService,
    create_analytics_service,
    create_analytics_cache
)

# خطاها
from .exceptions import (
    ClickHouseBaseError,
    ConnectionError,
    PoolExhaustedError,
    QueryError,
    SecurityError,
    OperationalError
)

logger = logging.getLogger(__name__)

logger.info("Initializing ClickHouse Module...")

__version__ = "1.0.0"

__all__ = [
    # کلاس‌های اصلی
    "ClickHouseConfig",

    # آداپتورها
    "ClickHouseConnectionPool",
    "CircuitBreaker",
    "RetryHandler",
    "ClickHouseLoadBalancer",
    "ClickHouseAdapter",
    "create_adapter",

    # دامنه
    "AnalyticsQuery",
    "AnalyticsResult",

    # یکپارچه‌سازی
    "GraphQLLayer",
    "RestAPI",
    "StreamProcessor",
    "create_graphql_layer",
    "create_rest_api",
    "create_stream_processor",

    # مدیریت
    "BackupManager",
    "DataLifecycleManager",
    "MigrationManager",
    "create_backup_manager",
    "create_lifecycle_manager",
    "create_migration_manager",

    # مانیتورینگ
    "PerformanceMonitor",
    "HealthCheck",
    "PrometheusExporter",
    "start_monitoring",

    # بهینه‌سازی
    "DataCompressor",
    "QueryOptimizer",

    # امنیت
    "AccessControl",
    "AuditLogger",
    "EncryptionManager",
    "create_encryption_manager",
    "create_access_control",
    "create_audit_logger",

    # سرویس‌ها
    "AnalyticsCache",
    "AnalyticsService",
    "create_analytics_service",
    "create_analytics_cache",

    # خطاها
    "ClickHouseBaseError",
    "ConnectionError",
    "QueryError",
    "SecurityError",
    "OperationalError"
]


def setup_clickhouse(custom_config: Optional[dict] = None):
    """
    راه‌اندازی و پیکربندی یکپارچه ClickHouse

    این تابع تمام ماژول‌های مورد نیاز ClickHouse را راه‌اندازی می‌کند
    و یک محیط آماده به کار برای شروع استفاده فراهم می‌کند.

    Args:
        custom_config (Optional[dict]): تنظیمات سفارشی (اختیاری)

    Returns:
        tuple: آداپتور، سرویس تحلیلی و نمونه GraphQL
    """
    # ایجاد و پیکربندی آداپتور
    adapter = create_adapter(custom_config)

    # راه‌اندازی سرویس‌های تحلیلی
    analytics_service = create_analytics_service(adapter)

    # راه‌اندازی مانیتورینگ
    start_monitoring()

    # راه‌اندازی GraphQL
    graphql_layer = create_graphql_layer(analytics_service)

    logger.info("ClickHouse setup completed successfully")

    return adapter, analytics_service, graphql_layer
