# مقداردهی اولیه ماژول TimescaleDB

# 🔹 تنظیمات و مدیریت اتصال
from .config import TimescaleDBConfig, ConnectionPool, ReadWriteSplitter

# 🔹 مدل‌های داده‌ای
from .domain import TimeSeriesData, TableSchema, TimeRange

# 🔹 مدیریت ذخیره‌سازی و اجرای کوئری‌ها
from .storage import TimescaleDBStorage

# 🔹 سرویس‌های بهینه‌سازی
from .optimization import CacheManager, QueryOptimizer, DataCompressor

# 🔹 مدیریت مقیاس‌پذیری
from .scaling import ReplicationManager, PartitionManager

# 🔹 ابزارهای امنیتی
from .security import AccessControl, AuditLog, EncryptionManager

# 🔹 سیستم مانیتورینگ
from .monitoring import MetricsCollector, SlowQueryAnalyzer, HealthCheck

# 🔹 سرویس‌های سطح بالا
from .service import DatabaseService, ContinuousAggregation, DataRetention

# 🔹 اسکریپت‌های مدیریتی
from .scripts import cleanup_old_data, create_backup, restore_backup, analyze_performance

# تعریف TimescaleDBService به عنوان alias برای DatabaseService
# برای حفظ سازگاری با واردسازی‌های فعلی در infrastructure/__init__.py
TimescaleDBService = DatabaseService

__all__ = [
    # تنظیمات
    "TimescaleDBConfig", "ConnectionPool", "ReadWriteSplitter",

    # مدل‌های داده‌ای
    "TimeSeriesData", "TableSchema", "TimeRange",

    # ذخیره‌سازی
    "TimescaleDBStorage",

    # بهینه‌سازی
    "CacheManager", "QueryOptimizer", "DataCompressor",

    # مقیاس‌پذیری
    "ReplicationManager", "PartitionManager",

    # امنیت
    "AccessControl", "AuditLog", "EncryptionManager",

    # مانیتورینگ
    "MetricsCollector", "SlowQueryAnalyzer", "HealthCheck",

    # سرویس‌های اصلی
    "DatabaseService", "ContinuousAggregation", "DataRetention",

    # alias برای سازگاری با infrastructure/__init__.py
    "TimescaleDBService",

    # اسکریپت‌های مدیریتی
    "cleanup_old_data", "create_backup", "restore_backup", "analyze_performance"
]