# infrastructure/clickhouse/management/migration_manager.py
"""
مدیریت مهاجرت‌های پایگاه داده برای ClickHouse
"""

import logging
import os
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from ..adapters.clickhouse_adapter import ClickHouseAdapter
from ..exceptions import DataManagementError, QueryError, OperationalError
from ..config.config import config
from ..security.audit_log import AuditLogger

logger = logging.getLogger(__name__)


class MigrationManager:
    """
    مدیریت مهاجرت‌های پایگاه داده برای ClickHouse

    این کلاس وظیفه مدیریت فرآیند مهاجرت پایگاه داده، از جمله ایجاد جداول جدید،
    تغییر ساختار جداول موجود و تغییرات دیگر در ClickHouse را بر عهده دارد.
    """

    def __init__(self, clickhouse_adapter: ClickHouseAdapter,
                 migrations_path: Optional[str] = None,
                 migrations_table: str = "migrations",
                 audit_logger: Optional[AuditLogger] = None):
        """
        مقداردهی اولیه مدیریت مهاجرت‌های پایگاه داده

        Args:
            clickhouse_adapter (ClickHouseAdapter): آداپتور اتصال به ClickHouse
            migrations_path (str, optional): مسیر دایرکتوری شامل فایل‌های مهاجرت
            migrations_table (str): نام جدول برای ثبت وضعیت مهاجرت‌ها
            audit_logger (AuditLogger, optional): لاگر رخدادهای امنیتی
        """
        self.clickhouse_adapter = clickhouse_adapter
        self.migrations_table = migrations_table

        # مسیر دایرکتوری مهاجرت‌ها
        if migrations_path is None:
            # استفاده از مسیر پیش‌فرض
            storage_root = os.getenv("STORAGE_ROOT", "./storage")
            migrations_path = os.path.join(storage_root, "migrations/clickhouse")
        self.migrations_path = migrations_path

        # آماده‌سازی لاگر امنیتی
        self.audit_logger = audit_logger
        if self.audit_logger is None:
            from ..security import create_audit_logger
            self.audit_logger = create_audit_logger(app_name="migration_manager")

        logger.info(f"Migration Manager initialized with migrations path: {self.migrations_path}")

    async def initialize(self) -> None:
        """
        آماده‌سازی زیرساخت مورد نیاز برای مدیریت مهاجرت‌ها

        این متد جدول مهاجرت‌ها را در صورت عدم وجود ایجاد می‌کند.

        Raises:
            DataManagementError: در صورت بروز خطا در آماده‌سازی
        """
        try:
            # ایجاد جدول مهاجرت‌ها اگر وجود نداشته باشد
            check_query = f"EXISTS TABLE {self.migrations_table}"
            check_result = await self.clickhouse_adapter.execute(check_query)

            if not check_result or not check_result[0]['exists']:
                create_query = f"""
                    CREATE TABLE IF NOT EXISTS {self.migrations_table} (
                        id String,
                        name String,
                        applied_at DateTime DEFAULT now(),
                        success UInt8,
                        error_message String DEFAULT '',
                        batch UInt32
                    ) ENGINE = MergeTree() ORDER BY (id, applied_at)
                """

                await self.clickhouse_adapter.execute(create_query)
                logger.info(f"Created migrations table: {self.migrations_table}")

            # ایجاد دایرکتوری مهاجرت‌ها اگر وجود نداشته باشد
            os.makedirs(self.migrations_path, exist_ok=True)

            logger.info("Migration system initialized successfully")

        except (QueryError, OperationalError) as e:
            # خطاهای داخلی را مستقیماً منتقل می‌کنیم
            logger.error(f"Failed to initialize migration system: {str(e)}")
            raise DataManagementError(
                message="Failed to initialize migration system",
                code="MIG001",
                operation="initialize",
                details={"error": str(e)}
            )
        except Exception as e:
            # سایر خطاها
            error_msg = f"Unexpected error during migration system initialization: {str(e)}"
            logger.error(error_msg)
            raise DataManagementError(
                message=error_msg,
                code="MIG002",
                operation="initialize",
                details={"error": str(e)}
            )

    async def apply_migration(self, migration_query: str, migration_id: Optional[str] = None,
                              migration_name: Optional[str] = None) -> bool:
        """
        اجرای یک مهاجرت در پایگاه داده ClickHouse

        Args:
            migration_query (str): دستور SQL مربوط به مهاجرت
            migration_id (str, optional): شناسه مهاجرت. اگر ارائه نشود، یک شناسه بر اساس زمان ایجاد می‌شود.
            migration_name (str, optional): نام توصیفی مهاجرت

        Returns:
            bool: نتیجه موفقیت اجرای مهاجرت

        Raises:
            DataManagementError: در صورت بروز خطا در اجرای مهاجرت
        """
        if not migration_id:
            migration_id = datetime.utcnow().strftime('%Y%m%d%H%M%S')

        if not migration_name:
            migration_name = f"migration_{migration_id}"

        try:
            # اطمینان از اینکه جدول مهاجرت‌ها وجود دارد
            await self.initialize()

            # اجرای کوئری مهاجرت
            await self.clickhouse_adapter.execute(migration_query)

            # ثبت مهاجرت موفق
            batch = await self._get_next_batch()

            insert_query = f"""
                INSERT INTO {self.migrations_table} 
                (id, name, applied_at, success, batch)
                VALUES (:id, :name, now(), 1, :batch)
            """

            insert_params = {
                "id": migration_id,
                "name": migration_name,
                "batch": batch
            }

            await self.clickhouse_adapter.execute(insert_query, insert_params)

            logger.info(f"Migration {migration_id} ({migration_name}) applied successfully")
            self._log_migration_event(migration_id, migration_name, "apply", True)

            return True

        except (QueryError, OperationalError) as e:
            # ثبت مهاجرت ناموفق
            try:
                batch = await self._get_next_batch()

                insert_query = f"""
                    INSERT INTO {self.migrations_table} 
                    (id, name, applied_at, success, error_message, batch)
                    VALUES (:id, :name, now(), 0, :error, :batch)
                """

                insert_params = {
                    "id": migration_id,
                    "name": migration_name,
                    "error": str(e),
                    "batch": batch
                }

                await self.clickhouse_adapter.execute(insert_query, insert_params)
            except Exception as log_error:
                logger.error(f"Failed to log failed migration: {str(log_error)}")

            # خطاهای داخلی را مستقیماً منتقل می‌کنیم
            logger.error(f"Failed to apply migration {migration_id} ({migration_name}): {str(e)}")
            self._log_migration_event(migration_id, migration_name, "apply", False, str(e))
            raise DataManagementError(
                message=f"Failed to apply migration {migration_id} ({migration_name})",
                code="MIG003",
                operation="apply",
                details={"error": str(e), "migration_id": migration_id, "migration_name": migration_name}
            )
        except Exception as e:
            # سایر خطاها
            error_msg = f"Unexpected error while applying migration {migration_id} ({migration_name}): {str(e)}"
            logger.error(error_msg)
            self._log_migration_event(migration_id, migration_name, "apply", False, str(e))
            raise DataManagementError(
                message=error_msg,
                code="MIG004",
                operation="apply",
                details={"error": str(e), "migration_id": migration_id, "migration_name": migration_name}
            )

    async def rollback_migration(self, rollback_query: str, migration_id: str) -> bool:
        """
        بازگردانی (Rollback) یک مهاجرت در پایگاه داده ClickHouse

        Args:
            rollback_query (str): دستور SQL برای بازگردانی تغییرات
            migration_id (str): شناسه مهاجرت برای بازگردانی

        Returns:
            bool: نتیجه موفقیت بازگردانی مهاجرت

        Raises:
            DataManagementError: در صورت بروز خطا در بازگردانی مهاجرت
        """
        try:
            # دریافت اطلاعات مهاجرت
            migration_info = await self.get_migration_info(migration_id)
            if not migration_info:
                error_msg = f"Migration {migration_id} not found in migrations table"
                logger.error(error_msg)
                raise DataManagementError(
                    message=error_msg,
                    code="MIG005",
                    operation="rollback",
                    details={"migration_id": migration_id}
                )

            migration_name = migration_info.get('name', 'unknown')

            # اجرای کوئری بازگردانی
            await self.clickhouse_adapter.execute(rollback_query)

            # حذف مهاجرت از جدول مهاجرت‌ها
            delete_query = f"""
                DELETE FROM {self.migrations_table}
                WHERE id = :id
            """

            await self.clickhouse_adapter.execute(delete_query, {"id": migration_id})

            logger.info(f"Migration {migration_id} ({migration_name}) rolled back successfully")
            self._log_migration_event(migration_id, migration_name, "rollback", True)

            return True

        except (QueryError, OperationalError) as e:
            # خطاهای داخلی را مستقیماً منتقل می‌کنیم
            logger.error(f"Failed to rollback migration {migration_id}: {str(e)}")
            self._log_migration_event(migration_id, migration_name if 'migration_name' in locals() else 'unknown',
                                      "rollback", False, str(e))
            raise DataManagementError(
                message=f"Failed to rollback migration {migration_id}",
                code="MIG006",
                operation="rollback",
                details={"error": str(e), "migration_id": migration_id}
            )
        except Exception as e:
            # سایر خطاها
            error_msg = f"Unexpected error while rolling back migration {migration_id}: {str(e)}"
            logger.error(error_msg)
            self._log_migration_event(migration_id, migration_name if 'migration_name' in locals() else 'unknown',
                                      "rollback", False, str(e))
            raise DataManagementError(
                message=error_msg,
                code="MIG007",
                operation="rollback",
                details={"error": str(e), "migration_id": migration_id}
            )

    async def get_migrations(self) -> List[Dict[str, Any]]:
        """
        دریافت لیست تمام مهاجرت‌های اجرا شده

        Returns:
            List[Dict[str, Any]]: لیست مهاجرت‌های اجرا شده با اطلاعات هر مهاجرت

        Raises:
            DataManagementError: در صورت بروز خطا در دریافت لیست مهاجرت‌ها
        """
        try:
            # اطمینان از اینکه جدول مهاجرت‌ها وجود دارد
            await self.initialize()

            query = f"""
                SELECT 
                    id,
                    name,
                    applied_at,
                    success,
                    error_message,
                    batch
                FROM {self.migrations_table}
                ORDER BY applied_at ASC
            """

            result = await self.clickhouse_adapter.execute(query)

            return result

        except (QueryError, OperationalError) as e:
            # خطاهای داخلی را مستقیماً منتقل می‌کنیم
            logger.error(f"Failed to get migrations list: {str(e)}")
            raise DataManagementError(
                message="Failed to get migrations list",
                code="MIG008",
                operation="get_migrations",
                details={"error": str(e)}
            )
        except Exception as e:
            # سایر خطاها
            error_msg = f"Unexpected error while getting migrations list: {str(e)}"
            logger.error(error_msg)
            raise DataManagementError(
                message=error_msg,
                code="MIG009",
                operation="get_migrations",
                details={"error": str(e)}
            )

    async def get_migration_info(self, migration_id: str) -> Optional[Dict[str, Any]]:
        """
        دریافت اطلاعات یک مهاجرت خاص

        Args:
            migration_id (str): شناسه مهاجرت

        Returns:
            Dict[str, Any] | None: اطلاعات مهاجرت یا None اگر یافت نشد

        Raises:
            DataManagementError: در صورت بروز خطا در دریافت اطلاعات مهاجرت
        """
        try:
            query = f"""
                SELECT 
                    id,
                    name,
                    applied_at,
                    success,
                    error_message,
                    batch
                FROM {self.migrations_table}
                WHERE id = :id
            """

            result = await self.clickhouse_adapter.execute(query, {"id": migration_id})

            if not result:
                return None

            return result[0]

        except (QueryError, OperationalError) as e:
            # خطاهای داخلی را مستقیماً منتقل می‌کنیم
            logger.error(f"Failed to get migration info for {migration_id}: {str(e)}")
            raise DataManagementError(
                message=f"Failed to get migration info for {migration_id}",
                code="MIG010",
                operation="get_migration_info",
                details={"error": str(e), "migration_id": migration_id}
            )
        except Exception as e:
            # سایر خطاها
            error_msg = f"Unexpected error while getting migration info for {migration_id}: {str(e)}"
            logger.error(error_msg)
            raise DataManagementError(
                message=error_msg,
                code="MIG011",
                operation="get_migration_info",
                details={"error": str(e), "migration_id": migration_id}
            )

    async def run_migrations(self, migrations_folder: Optional[str] = None) -> Tuple[int, int]:
        """
        اجرای تمام مهاجرت‌های جدید از پوشه مهاجرت‌ها

        Args:
            migrations_folder (str, optional): مسیر پوشه شامل فایل‌های مهاجرت.
                اگر ارائه نشود، از مسیر پیش‌فرض استفاده می‌شود.

        Returns:
            Tuple[int, int]: تعداد مهاجرت‌های موفق و ناموفق

        Raises:
            DataManagementError: در صورت بروز خطا در اجرای مهاجرت‌ها
        """
        folder_path = migrations_folder or self.migrations_path

        try:
            # اطمینان از اینکه جدول مهاجرت‌ها وجود دارد
            await self.initialize()

            # دریافت لیست مهاجرت‌های اجرا شده
            executed_migrations = await self.get_migrations()
            executed_ids = {m['id'] for m in executed_migrations}

            # اطمینان از وجود پوشه مهاجرت‌ها
            if not os.path.exists(folder_path):
                error_msg = f"Migrations folder does not exist: {folder_path}"
                logger.error(error_msg)
                raise DataManagementError(
                    message=error_msg,
                    code="MIG012",
                    operation="run_migrations",
                    details={"folder_path": folder_path}
                )

            # اسکن فایل‌های مهاجرت و اجرای موارد جدید
            migration_files = []
            for file in os.listdir(folder_path):
                if file.endswith('.sql') or file.endswith('.json'):
                    migration_id = file.split('_')[0]
                    if migration_id not in executed_ids:
                        migration_files.append((migration_id, file))

            # مرتب‌سازی بر اساس شناسه مهاجرت
            migration_files.sort()

            successful = 0
            failed = 0

            # اجرای مهاجرت‌های جدید
            for migration_id, file in migration_files:
                file_path = os.path.join(folder_path, file)
                migration_name = file.replace('.sql', '').replace('.json', '')

                try:
                    # خواندن محتوای فایل مهاجرت
                    with open(file_path, 'r') as f:
                        content = f.read()

                    # پردازش فایل‌های JSON برای استخراج کوئری
                    if file.endswith('.json'):
                        migration_data = json.loads(content)
                        migration_query = migration_data.get('up', '')
                        if not migration_query:
                            logger.warning(f"Migration file {file} does not contain 'up' key")
                            continue
                    else:
                        migration_query = content

                    # اجرای مهاجرت
                    success = await self.apply_migration(migration_query, migration_id, migration_name)

                    if success:
                        successful += 1
                    else:
                        failed += 1

                except Exception as e:
                    failed += 1
                    logger.error(f"Failed to apply migration from file {file}: {str(e)}")
                    self._log_migration_event(migration_id, migration_name, "apply", False, str(e))

            logger.info(f"Ran {successful + failed} migrations: {successful} successful, {failed} failed")
            return successful, failed

        except Exception as e:
            # خطای کلی در اجرای مهاجرت‌ها
            error_msg = f"Failed to run migrations from folder {folder_path}: {str(e)}"
            logger.error(error_msg)
            raise DataManagementError(
                message=error_msg,
                code="MIG013",
                operation="run_migrations",
                details={"error": str(e), "folder_path": folder_path}
            )

    async def _get_next_batch(self) -> int:
        """
        دریافت شماره دسته بعدی برای مهاجرت‌ها

        Returns:
            int: شماره دسته بعدی
        """
        try:
            query = f"""
                SELECT MAX(batch) as last_batch
                FROM {self.migrations_table}
            """

            result = await self.clickhouse_adapter.execute(query)

            last_batch = result[0]['last_batch'] if result and result[0]['last_batch'] else 0
            return last_batch + 1

        except Exception as e:
            logger.warning(f"Error getting next batch number: {str(e)}, using 1")
            return 1

    def _log_migration_event(self, migration_id: str, migration_name: str, operation: str,
                             success: bool, error_message: Optional[str] = None):
        """
        ثبت لاگ برای عملیات‌های مهاجرت

        Args:
            migration_id (str): شناسه مهاجرت
            migration_name (str): نام مهاجرت
            operation (str): نوع عملیات (apply/rollback)
            success (bool): موفقیت یا شکست عملیات
            error_message (str, optional): پیام خطا در صورت شکست
        """
        if self.audit_logger:
            details = {
                "migration_id": migration_id,
                "migration_name": migration_name,
                "operation": operation
            }

            if error_message:
                details["error"] = error_message

            status = "success" if success else "failure"

            self.audit_logger.log_event(
                username="migration_manager",
                action=f"migration_{operation}",
                status=status,
                details=str(details),
                resource=f"migration_{migration_id}"
            )
