# infrastructure/clickhouse/management/data_lifecycle.py
"""
مدیریت چرخه عمر داده‌ها در ClickHouse برای حفظ عملکرد و مدیریت بهینه فضا
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from ..adapters.clickhouse_adapter import ClickHouseAdapter
from ..exceptions import DataManagementError, QueryError, OperationalError
from ..config.config import config
from ..security.audit_log import AuditLogger

logger = logging.getLogger(__name__)


class DataLifecycleManager:
    """
    مدیریت چرخه عمر داده‌ها در ClickHouse برای حفظ عملکرد و مدیریت بهینه فضا

    این کلاس وظیفه مدیریت و بهینه‌سازی داده‌ها در طول چرخه عمرشان را بر عهده دارد،
    شامل حذف داده‌های منقضی‌شده، فشرده‌سازی و بازسازی جداول.
    """

    def __init__(self, clickhouse_adapter: ClickHouseAdapter, retention_days: Optional[int] = None,
                 audit_logger: Optional[AuditLogger] = None):
        """
        مقداردهی اولیه مدیریت داده‌ها

        Args:
            clickhouse_adapter (ClickHouseAdapter): آداپتور اتصال به ClickHouse
            retention_days (int, optional): تعداد روزهایی که داده‌ها قبل از حذف نگهداری می‌شوند
            audit_logger (AuditLogger, optional): لاگر رخدادهای امنیتی
        """
        self.clickhouse_adapter = clickhouse_adapter

        # استفاده از تنظیمات متمرکز اگر تعداد روزها ارائه نشده باشد
        if retention_days is None:
            retention_days = config.get_data_management_config()["retention_days"]
        self.retention_days = retention_days

        # آماده‌سازی لاگر امنیتی
        self.audit_logger = audit_logger
        if self.audit_logger is None:
            from ..security import create_audit_logger
            self.audit_logger = create_audit_logger(app_name="data_lifecycle")

        logger.info(f"Data Lifecycle Manager initialized with retention period of {self.retention_days} days")

    async def delete_expired_data(self, table_name: str, date_column: str = "created_at") -> int:
        """
        حذف داده‌های منقضی‌شده از یک جدول خاص بر اساس تاریخ انقضا

        Args:
            table_name (str): نام جدول
            date_column (str): نام ستون تاریخ برای بررسی انقضا

        Returns:
            int: تعداد رکوردهای حذف شده

        Raises:
            DataManagementError: در صورت بروز خطا در عملیات حذف
        """
        # محاسبه تاریخ انقضا
        expiration_date = datetime.utcnow() - timedelta(days=self.retention_days)
        expiration_date_str = expiration_date.strftime('%Y-%m-%d')

        try:
            # ابتدا تعداد رکوردهای منقضی را بررسی می‌کنیم
            count_query = "SELECT COUNT(*) as expired_count FROM :table WHERE :date_column < :expiration_date"
            count_params = {
                "table": table_name,
                "date_column": date_column,
                "expiration_date": expiration_date_str
            }

            count_result = await self.clickhouse_adapter.execute(count_query, count_params)
            expired_count = count_result[0]['expired_count'] if count_result and count_result[0] else 0

            # اگر داده‌ای برای حذف نیست، برمی‌گردیم
            if expired_count == 0:
                logger.info(f"No expired data found in table {table_name}")
                self._log_lifecycle_event(table_name, "delete_expired", True, 0)
                return 0

            # حذف داده‌های منقضی
            delete_query = "ALTER TABLE :table DELETE WHERE :date_column < :expiration_date"

            await self.clickhouse_adapter.execute(delete_query, count_params)

            logger.info(
                f"Deleted {expired_count} expired records from {table_name} where {date_column} < {expiration_date_str}")
            self._log_lifecycle_event(table_name, "delete_expired", True, expired_count)

            return expired_count

        except (QueryError, OperationalError) as e:
            # خطاهای داخلی را مستقیماً منتقل می‌کنیم
            logger.error(f"Failed to delete expired data from {table_name}: {str(e)}")
            self._log_lifecycle_event(table_name, "delete_expired", False, 0, str(e))
            raise DataManagementError(
                message=f"Failed to delete expired data from {table_name}",
                code="LCM001",
                operation="delete_expired",
                table_name=table_name,
                details={"error": str(e), "date_column": date_column, "retention_days": self.retention_days}
            )
        except Exception as e:
            # سایر خطاها
            error_msg = f"Unexpected error during deletion of expired data from {table_name}: {str(e)}"
            logger.error(error_msg)
            self._log_lifecycle_event(table_name, "delete_expired", False, 0, str(e))
            raise DataManagementError(
                message=error_msg,
                code="LCM002",
                operation="delete_expired",
                table_name=table_name,
                details={"error": str(e), "date_column": date_column, "retention_days": self.retention_days}
            )

    async def optimize_table(self, table_name: str, final: bool = False, deduplicate: bool = False) -> bool:
        """
        بهینه‌سازی و فشرده‌سازی جدول داده

        Args:
            table_name (str): نام جدول
            final (bool): آیا بهینه‌سازی FINAL انجام شود
            deduplicate (bool): آیا حذف داده‌های تکراری انجام شود

        Returns:
            bool: نتیجه موفقیت عملیات بهینه‌سازی

        Raises:
            DataManagementError: در صورت بروز خطا در عملیات بهینه‌سازی
        """
        try:
            # ساخت کوئری بهینه‌سازی
            optimize_query = "OPTIMIZE TABLE :table"
            params = {"table": table_name}

            if final:
                optimize_query += " FINAL"

            if deduplicate:
                optimize_query += " DEDUPLICATE"

            await self.clickhouse_adapter.execute(optimize_query, params)

            logger.info(f"Table {table_name} optimized successfully (final={final}, deduplicate={deduplicate})")
            self._log_lifecycle_event(table_name, "optimize", True, 0)

            return True

        except (QueryError, OperationalError) as e:
            # خطاهای داخلی را مستقیماً منتقل می‌کنیم
            logger.error(f"Failed to optimize table {table_name}: {str(e)}")
            self._log_lifecycle_event(table_name, "optimize", False, 0, str(e))
            raise DataManagementError(
                message=f"Failed to optimize table {table_name}",
                code="LCM003",
                operation="optimize",
                table_name=table_name,
                details={"error": str(e), "final": final, "deduplicate": deduplicate}
            )
        except Exception as e:
            # سایر خطاها
            error_msg = f"Unexpected error during optimization of table {table_name}: {str(e)}"
            logger.error(error_msg)
            self._log_lifecycle_event(table_name, "optimize", False, 0, str(e))
            raise DataManagementError(
                message=error_msg,
                code="LCM004",
                operation="optimize",
                table_name=table_name,
                details={"error": str(e), "final": final, "deduplicate": deduplicate}
            )

    async def get_table_size_info(self, table_name: str) -> Dict[str, Any]:
        """
        دریافت اطلاعات اندازه و تعداد رکوردهای یک جدول

        Args:
            table_name (str): نام جدول

        Returns:
            Dict[str, Any]: اطلاعات اندازه و تعداد رکوردها

        Raises:
            DataManagementError: در صورت بروز خطا در دریافت اطلاعات
        """
        try:
            # دریافت تعداد رکوردها
            count_query = "SELECT COUNT(*) as row_count FROM :table"
            count_result = await self.clickhouse_adapter.execute(count_query, {"table": table_name})
            row_count = count_result[0]['row_count'] if count_result else 0

            # دریافت اطلاعات اندازه
            size_query = """
                SELECT 
                    table,
                    formatReadableSize(sum(bytes)) as size,
                    sum(bytes) as bytes,
                    sum(rows) as rows,
                    max(modification_time) as latest_modification,
                    min(min_date) as min_date,
                    max(max_date) as max_date
                FROM system.parts
                WHERE active AND table = :table
                GROUP BY table
            """

            size_result = await self.clickhouse_adapter.execute(size_query, {"table": table_name})

            if not size_result:
                return {
                    "table_name": table_name,
                    "row_count": row_count,
                    "size_bytes": 0,
                    "readable_size": "0 B",
                    "latest_modification": None
                }

            table_info = {
                "table_name": table_name,
                "row_count": row_count,
                "size_bytes": size_result[0].get('bytes', 0),
                "readable_size": size_result[0].get('size', '0 B'),
                "latest_modification": size_result[0].get('latest_modification')
            }

            # اضافه کردن اطلاعات تاریخ اگر موجود باشند
            if 'min_date' in size_result[0] and size_result[0]['min_date']:
                table_info["min_date"] = size_result[0]['min_date']

            if 'max_date' in size_result[0] and size_result[0]['max_date']:
                table_info["max_date"] = size_result[0]['max_date']

            return table_info

        except (QueryError, OperationalError) as e:
            # خطاهای داخلی را مستقیماً منتقل می‌کنیم
            logger.error(f"Failed to get size info for table {table_name}: {str(e)}")
            raise DataManagementError(
                message=f"Failed to get size info for table {table_name}",
                code="LCM005",
                operation="get_size_info",
                table_name=table_name,
                details={"error": str(e)}
            )
        except Exception as e:
            # سایر خطاها
            error_msg = f"Unexpected error while getting size info for table {table_name}: {str(e)}"
            logger.error(error_msg)
            raise DataManagementError(
                message=error_msg,
                code="LCM006",
                operation="get_size_info",
                table_name=table_name,
                details={"error": str(e)}
            )

    async def analyze_database_size(self) -> List[Dict[str, Any]]:
        """
        تحلیل اندازه تمام جداول پایگاه داده

        Returns:
            List[Dict[str, Any]]: لیست جداول با اطلاعات اندازه، مرتب‌شده بر اساس اندازه

        Raises:
            DataManagementError: در صورت بروز خطا در تحلیل پایگاه داده
        """
        try:
            query = """
                SELECT 
                    table,
                    database,
                    formatReadableSize(sum(bytes)) as size,
                    sum(bytes) as bytes,
                    sum(rows) as rows,
                    max(modification_time) as latest_modification
                FROM system.parts
                WHERE active
                GROUP BY database, table
                ORDER BY bytes DESC
            """

            result = await self.clickhouse_adapter.execute(query)

            # تبدیل نتیجه به فرمت مناسب
            table_stats = []
            for row in result:
                table_stats.append({
                    "table_name": row.get('table', ''),
                    "database": row.get('database', ''),
                    "row_count": row.get('rows', 0),
                    "size_bytes": row.get('bytes', 0),
                    "readable_size": row.get('size', '0 B'),
                    "latest_modification": row.get('latest_modification')
                })

            return table_stats

        except (QueryError, OperationalError) as e:
            # خطاهای داخلی را مستقیماً منتقل می‌کنیم
            logger.error(f"Failed to analyze database size: {str(e)}")
            raise DataManagementError(
                message="Failed to analyze database size",
                code="LCM007",
                operation="analyze_db_size",
                details={"error": str(e)}
            )
        except Exception as e:
            # سایر خطاها
            error_msg = f"Unexpected error during database size analysis: {str(e)}"
            logger.error(error_msg)
            raise DataManagementError(
                message=error_msg,
                code="LCM008",
                operation="analyze_db_size",
                details={"error": str(e)}
            )

    def _log_lifecycle_event(self, table_name: str, operation: str, success: bool,
                             affected_rows: int = 0, error_message: Optional[str] = None):
        """
        ثبت لاگ برای عملیات‌های چرخه عمر داده‌ها

        Args:
            table_name (str): نام جدول
            operation (str): نوع عملیات
            success (bool): موفقیت یا شکست عملیات
            affected_rows (int): تعداد رکوردهای تحت تأثیر
            error_message (str, optional): پیام خطا در صورت شکست
        """
        if self.audit_logger:
            details = {
                "table_name": table_name,
                "operation": operation,
                "affected_rows": affected_rows
            }

            if error_message:
                details["error"] = error_message

            status = "success" if success else "failure"

            self.audit_logger.log_event(
                username="data_lifecycle_manager",
                action=f"lifecycle_{operation}",
                status=status,
                details=str(details),
                resource=table_name
            )
