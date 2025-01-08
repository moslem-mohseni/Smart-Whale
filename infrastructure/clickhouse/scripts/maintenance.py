# infrastructure/clickhouse/scripts/maintenance.py

"""
این ماژول شامل اسکریپت‌های مدیریتی و نگهداری برای ClickHouse است.
این اسکریپت‌ها وظایفی مانند بهینه‌سازی جداول، پاکسازی داده‌های قدیمی،
تهیه نسخه پشتیبان و بررسی سلامت سیستم را انجام می‌دهند.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from ..adapters.clickhouse_adapter import ClickHouseAdapter
from ..service.analytics_service import AnalyticsService

logger = logging.getLogger(__name__)


class MaintenanceManager:
    """مدیریت عملیات نگهداری ClickHouse"""

    def __init__(self, analytics_service: AnalyticsService):
        self.analytics_service = analytics_service
        self._adapter = analytics_service._adapter

    async def optimize_tables(self, table_names: Optional[list[str]] = None) -> None:
        """
        بهینه‌سازی جداول ClickHouse

        Args:
            table_names: لیست جداول مورد نظر (اگر None باشد، همه جداول بهینه می‌شوند)
        """
        if table_names is None:
            # دریافت لیست تمام جداول
            result = await self._adapter.execute_query(
                "SHOW TABLES FROM system"
            )
            table_names = [row['name'] for row in result]

        for table in table_names:
            try:
                await self._adapter.execute_query(f"OPTIMIZE TABLE {table} FINAL")
                logger.info(f"Optimized table: {table}")
            except Exception as e:
                logger.error(f"Failed to optimize table {table}: {str(e)}")

    async def cleanup_old_data(self, days: int = 90) -> None:
        """
        پاکسازی داده‌های قدیمی

        Args:
            days: داده‌های قدیمی‌تر از این تعداد روز حذف می‌شوند
        """
        cutoff_date = datetime.now() - timedelta(days=days)

        try:
            # حذف داده‌های قدیمی از جدول events
            await self._adapter.execute_query(
                "ALTER TABLE events DELETE WHERE timestamp < %(cutoff_date)s",
                {'cutoff_date': cutoff_date}
            )
            logger.info(f"Cleaned up data older than {days} days")

        except Exception as e:
            logger.error(f"Failed to cleanup old data: {str(e)}")

    async def check_system_health(self) -> Dict[str, Any]:
        """
        بررسی سلامت سیستم

        Returns:
            دیکشنری حاوی وضعیت سلامت سیستم
        """
        health_checks = {
            'database_connection': True,
            'table_states': {},
            'disk_usage': {},
            'performance_metrics': {}
        }

        try:
            # بررسی وضعیت جداول
            tables = await self._adapter.execute_query("""
                SELECT
                    table,
                    total_rows,
                    total_bytes
                FROM system.tables
                WHERE database = currentDatabase()
            """)

            for table in tables:
                health_checks['table_states'][table['table']] = {
                    'rows': table['total_rows'],
                    'size': table['total_bytes']
                }

            # بررسی فضای دیسک
            disk_usage = await self._adapter.execute_query("""
                SELECT
                    name,
                    free_space,
                    total_space
                FROM system.disks
            """)

            for disk in disk_usage:
                health_checks['disk_usage'][disk['name']] = {
                    'free_space': disk['free_space'],
                    'total_space': disk['total_space'],
                    'usage_percent': (
                            (disk['total_space'] - disk['free_space']) /
                            disk['total_space'] * 100
                    )
                }

            # بررسی متریک‌های عملکردی
            metrics = await self._adapter.execute_query("""
                SELECT
                    metric,
                    value
                FROM system.metrics
                WHERE metric IN (
                    'Query',
                    'QueryThread',
                    'QueryPreempted',
                    'TCPConnection'
                )
            """)

            for metric in metrics:
                health_checks['performance_metrics'][metric['metric']] = metric['value']

        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            health_checks['database_connection'] = False

        return health_checks

    async def backup_table(self, table_name: str, backup_path: str) -> None:
        """
        تهیه نسخه پشتیبان از یک جدول

        Args:
            table_name: نام جدول
            backup_path: مسیر ذخیره‌سازی نسخه پشتیبان
        """
        try:
            await self._adapter.execute_query(f"""
                BACKUP TABLE {table_name}
                TO {backup_path}
            """)
            logger.info(f"Backed up table {table_name} to {backup_path}")

        except Exception as e:
            logger.error(f"Failed to backup table {table_name}: {str(e)}")
            raise