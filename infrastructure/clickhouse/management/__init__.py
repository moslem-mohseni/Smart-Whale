# infrastructure/clickhouse/management/__init__.py
"""
ماژول مدیریت ClickHouse

این ماژول شامل کلاس‌ها و توابع مربوط به مدیریت ClickHouse است:
- BackupManager: مدیریت فرآیند پشتیبان‌گیری از داده‌ها
- DataLifecycleManager: مدیریت چرخه عمر داده‌ها و حذف داده‌های منقضی‌شده
- MigrationManager: مدیریت مهاجرت‌های پایگاه داده
"""

import logging
from .backup_manager import BackupManager
from .data_lifecycle import DataLifecycleManager
from .migration_manager import MigrationManager
from ..adapters.clickhouse_adapter import ClickHouseAdapter
from ..config.config import config

logger = logging.getLogger(__name__)

logger.info("Initializing ClickHouse Management Module...")

__all__ = [
    "BackupManager",
    "DataLifecycleManager",
    "MigrationManager",
    "create_backup_manager",
    "create_lifecycle_manager",
    "create_migration_manager"
]


def create_backup_manager(clickhouse_adapter: ClickHouseAdapter = None, backup_dir: str = None):
    """
    ایجاد یک نمونه از BackupManager با تنظیمات مناسب

    Args:
        clickhouse_adapter (ClickHouseAdapter, optional): آداپتور اتصال به ClickHouse
        backup_dir (str, optional): مسیر پوشه ذخیره پشتیبان‌ها

    Returns:
        BackupManager: نمونه آماده استفاده از BackupManager
    """
    if clickhouse_adapter is None:
        from ..adapters import create_adapter
        clickhouse_adapter = create_adapter()

    if backup_dir is None:
        backup_dir = config.get_data_management_config()["backup_dir"]

    return BackupManager(clickhouse_adapter=clickhouse_adapter, backup_dir=backup_dir)


def create_lifecycle_manager(clickhouse_adapter: ClickHouseAdapter = None, retention_days: int = None):
    """
    ایجاد یک نمونه از DataLifecycleManager با تنظیمات مناسب

    Args:
        clickhouse_adapter (ClickHouseAdapter, optional): آداپتور اتصال به ClickHouse
        retention_days (int, optional): تعداد روزهای نگهداری داده‌ها

    Returns:
        DataLifecycleManager: نمونه آماده استفاده از DataLifecycleManager
    """
    if clickhouse_adapter is None:
        from ..adapters import create_adapter
        clickhouse_adapter = create_adapter()

    if retention_days is None:
        retention_days = config.get_data_management_config()["retention_days"]

    return DataLifecycleManager(clickhouse_adapter=clickhouse_adapter, retention_days=retention_days)


def create_migration_manager(clickhouse_adapter: ClickHouseAdapter = None):
    """
    ایجاد یک نمونه از MigrationManager با تنظیمات مناسب

    Args:
        clickhouse_adapter (ClickHouseAdapter, optional): آداپتور اتصال به ClickHouse

    Returns:
        MigrationManager: نمونه آماده استفاده از MigrationManager
    """
    if clickhouse_adapter is None:
        from ..adapters import create_adapter
        clickhouse_adapter = create_adapter()

    return MigrationManager(clickhouse_adapter=clickhouse_adapter)
