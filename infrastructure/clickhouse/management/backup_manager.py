# infrastructure/clickhouse/management/backup_manager.py
"""
مدیریت فرآیند پشتیبان‌گیری از داده‌های ClickHouse
"""

import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any, List
from ..adapters.clickhouse_adapter import ClickHouseAdapter
from ..exceptions import BackupError, QueryError, OperationalError
from ..config.config import config
from ..security.audit_log import AuditLogger

logger = logging.getLogger(__name__)


class BackupManager:
    """
    مدیریت فرآیند پشتیبان‌گیری از داده‌های ClickHouse

    این کلاس وظیفه ایجاد و بازیابی پشتیبان از داده‌های ClickHouse را بر عهده دارد.
    پشتیبان‌گیری‌ها به صورت فایل SQL ذخیره می‌شوند.
    """

    def __init__(self, clickhouse_adapter: ClickHouseAdapter, backup_dir: Optional[str] = None,
                 audit_logger: Optional[AuditLogger] = None):
        """
        مقداردهی اولیه مدیریت پشتیبان‌گیری

        Args:
            clickhouse_adapter (ClickHouseAdapter): آداپتور ClickHouse برای اجرای دستورات
            backup_dir (str, optional): مسیر ذخیره‌سازی فایل‌های پشتیبان
            audit_logger (AuditLogger, optional): لاگر رخدادهای امنیتی
        """
        self.clickhouse_adapter = clickhouse_adapter

        # استفاده از تنظیمات متمرکز اگر مسیر پشتیبان ارائه نشده باشد
        if backup_dir is None:
            backup_dir = config.get_data_management_config()["backup_dir"]
        self.backup_dir = backup_dir

        # آماده‌سازی لاگر امنیتی
        self.audit_logger = audit_logger
        if self.audit_logger is None:
            from ..security import create_audit_logger
            self.audit_logger = create_audit_logger(app_name="backup_manager")

        # اطمینان از وجود دایرکتوری پشتیبان
        try:
            os.makedirs(self.backup_dir, exist_ok=True)
            logger.info(f"Backup directory initialized: {self.backup_dir}")
        except Exception as e:
            error_msg = f"Failed to create backup directory {self.backup_dir}: {str(e)}"
            logger.error(error_msg)
            raise BackupError(
                message=error_msg,
                code="BKP001",
                operation="directory_init"
            )

    async def create_backup(self, table_name: str, partition: Optional[str] = None) -> str:
        """
        ایجاد یک نسخه پشتیبان از جدول داده‌ها

        Args:
            table_name (str): نام جدول موردنظر برای پشتیبان‌گیری
            partition (str, optional): پارتیشن خاص برای پشتیبان‌گیری

        Returns:
            str: مسیر فایل پشتیبان ایجاد شده

        Raises:
            BackupError: در صورت بروز خطا در فرآیند پشتیبان‌گیری
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup_file_name = f"{table_name}_backup_{timestamp}.sql"
        backup_file_path = os.path.join(self.backup_dir, backup_file_name)

        try:
            # ساخت کوئری پشتیبان‌گیری با پارامترها
            params = {"table": table_name, "file_path": backup_file_path}
            query = "BACKUP TABLE :table TO Disk(:file_path)"

            # اضافه کردن پارتیشن به کوئری اگر مشخص شده باشد
            if partition:
                query += " PARTITION :partition"
                params["partition"] = partition

            # اجرای دستور پشتیبان‌گیری
            await self.clickhouse_adapter.execute(query, params)

            # لاگ موفقیت
            logger.info(f"Backup created successfully: {backup_file_path}")
            self._log_backup_event(table_name, "create", True, backup_file_path, partition)

            return backup_file_path

        except (QueryError, OperationalError) as e:
            # خطاهای داخلی را مستقیماً منتقل می‌کنیم
            logger.error(f"Failed to create backup for table {table_name}: {str(e)}")
            self._log_backup_event(table_name, "create", False, backup_file_path, partition, str(e))
            raise BackupError(
                message=f"Failed to create backup for table {table_name}",
                code="BKP002",
                operation="backup",
                table_name=table_name,
                backup_file=backup_file_path,
                details={"error": str(e), "partition": partition}
            )
        except Exception as e:
            # سایر خطاها
            error_msg = f"Unexpected error during backup of table {table_name}: {str(e)}"
            logger.error(error_msg)
            self._log_backup_event(table_name, "create", False, backup_file_path, partition, str(e))
            raise BackupError(
                message=error_msg,
                code="BKP003",
                operation="backup",
                table_name=table_name,
                backup_file=backup_file_path,
                details={"error": str(e), "partition": partition}
            )

    async def restore_backup(self, table_name: str, backup_file: str) -> bool:
        """
        بازیابی داده‌ها از یک فایل پشتیبان

        Args:
            table_name (str): نام جدول موردنظر برای بازیابی
            backup_file (str): مسیر فایل پشتیبان برای بازیابی داده‌ها

        Returns:
            bool: نتیجه موفقیت عملیات بازیابی

        Raises:
            BackupError: در صورت بروز خطا در فرآیند بازیابی
        """
        # اطمینان از وجود فایل پشتیبان
        backup_file_path = os.path.join(self.backup_dir, backup_file) if not os.path.dirname(
            backup_file) else backup_file
        if not os.path.exists(backup_file_path):
            error_msg = f"Backup file does not exist: {backup_file_path}"
            logger.error(error_msg)
            self._log_backup_event(table_name, "restore", False, backup_file_path, None, error_msg)
            raise BackupError(
                message=error_msg,
                code="BKP004",
                operation="restore",
                table_name=table_name,
                backup_file=backup_file_path
            )

        try:
            # استفاده از پارامترهای کوئری برای افزایش امنیت
            params = {"table": table_name, "file_path": backup_file_path}
            query = "RESTORE TABLE :table FROM Disk(:file_path)"

            # اجرای دستور بازیابی
            await self.clickhouse_adapter.execute(query, params)

            # لاگ موفقیت
            logger.info(f"Backup restored successfully from {backup_file_path} to table {table_name}")
            self._log_backup_event(table_name, "restore", True, backup_file_path)

            return True

        except (QueryError, OperationalError) as e:
            # خطاهای داخلی را مستقیماً منتقل می‌کنیم
            logger.error(f"Failed to restore backup for table {table_name}: {str(e)}")
            self._log_backup_event(table_name, "restore", False, backup_file_path, None, str(e))
            raise BackupError(
                message=f"Failed to restore backup for table {table_name}",
                code="BKP005",
                operation="restore",
                table_name=table_name,
                backup_file=backup_file_path,
                details={"error": str(e)}
            )
        except Exception as e:
            # سایر خطاها
            error_msg = f"Unexpected error during restore of table {table_name}: {str(e)}"
            logger.error(error_msg)
            self._log_backup_event(table_name, "restore", False, backup_file_path, None, str(e))
            raise BackupError(
                message=error_msg,
                code="BKP006",
                operation="restore",
                table_name=table_name,
                backup_file=backup_file_path,
                details={"error": str(e)}
            )

    async def list_backups(self, table_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        لیست کردن فایل‌های پشتیبان موجود

        Args:
            table_name (str, optional): فیلتر کردن لیست بر اساس نام جدول

        Returns:
            List[Dict[str, Any]]: لیست فایل‌های پشتیبان با اطلاعات هر فایل
        """
        try:
            backups = []

            for file in os.listdir(self.backup_dir):
                if file.endswith(".sql"):
                    # اگر نام جدول مشخص شده باشد و فایل مربوط به آن جدول نباشد، رد می‌کنیم
                    if table_name and not file.startswith(f"{table_name}_backup_"):
                        continue

                    file_path = os.path.join(self.backup_dir, file)
                    file_info = os.stat(file_path)

                    backup_info = {
                        "file_name": file,
                        "file_path": file_path,
                        "size_bytes": file_info.st_size,
                        "creation_time": datetime.fromtimestamp(file_info.st_ctime).isoformat(),
                        "table_name": file.split("_backup_")[0] if "_backup_" in file else "unknown"
                    }

                    backups.append(backup_info)

            logger.info(f"Listed {len(backups)} backup files")
            return backups

        except Exception as e:
            error_msg = f"Failed to list backup files: {str(e)}"
            logger.error(error_msg)
            raise BackupError(
                message=error_msg,
                code="BKP007",
                operation="list",
                details={"error": str(e), "table_filter": table_name}
            )

    async def delete_backup(self, backup_file: str) -> bool:
        """
        حذف یک فایل پشتیبان

        Args:
            backup_file (str): نام یا مسیر فایل پشتیبان برای حذف

        Returns:
            bool: نتیجه موفقیت عملیات حذف

        Raises:
            BackupError: در صورت بروز خطا در حذف فایل پشتیبان
        """
        # تشخیص مسیر کامل فایل
        backup_file_path = os.path.join(self.backup_dir, backup_file) if not os.path.dirname(
            backup_file) else backup_file

        # استخراج نام جدول برای لاگینگ
        table_name = "unknown"
        try:
            file_name = os.path.basename(backup_file_path)
            if "_backup_" in file_name:
                table_name = file_name.split("_backup_")[0]
        except Exception:
            pass

        try:
            # بررسی وجود فایل
            if not os.path.exists(backup_file_path):
                error_msg = f"Backup file does not exist: {backup_file_path}"
                logger.warning(error_msg)
                return False

            # حذف فایل
            os.remove(backup_file_path)

            logger.info(f"Backup file deleted: {backup_file_path}")
            self._log_backup_event(table_name, "delete", True, backup_file_path)

            return True

        except Exception as e:
            error_msg = f"Failed to delete backup file {backup_file_path}: {str(e)}"
            logger.error(error_msg)
            self._log_backup_event(table_name, "delete", False, backup_file_path, None, str(e))
            raise BackupError(
                message=error_msg,
                code="BKP008",
                operation="delete",
                backup_file=backup_file_path,
                table_name=table_name,
                details={"error": str(e)}
            )

    def _log_backup_event(self, table_name: str, operation: str, success: bool,
                          file_path: str, partition: Optional[str] = None,
                          error_message: Optional[str] = None):
        """
        ثبت لاگ برای عملیات‌های پشتیبان‌گیری

        Args:
            table_name (str): نام جدول
            operation (str): نوع عملیات (create/restore/delete)
            success (bool): موفقیت یا شکست عملیات
            file_path (str): مسیر فایل پشتیبان
            partition (str, optional): پارتیشن مورد استفاده
            error_message (str, optional): پیام خطا در صورت شکست
        """
        if self.audit_logger:
            details = {
                "table_name": table_name,
                "operation": operation,
                "backup_file": file_path
            }

            if partition:
                details["partition"] = partition

            if error_message:
                details["error"] = error_message

            status = "success" if success else "failure"

            self.audit_logger.log_event(
                username="backup_manager",
                action=f"backup_{operation}",
                status=status,
                details=str(details),
                resource=table_name
            )
