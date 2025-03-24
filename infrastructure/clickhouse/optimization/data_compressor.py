# infrastructure/clickhouse/optimization/data_compressor.py
import logging
from typing import Optional, Dict, Any, List
from ..config.config import config
from ..adapters.clickhouse_adapter import ClickHouseAdapter
from ..exceptions import (
    OperationalError, QueryError, DataManagementError
)

logger = logging.getLogger(__name__)


class DataCompressor:
    """
    فشرده‌سازی داده‌های ClickHouse برای بهینه‌سازی فضای ذخیره‌سازی

    این کلاس امکان اجرای عملیات بهینه‌سازی و فشرده‌سازی داده‌ها در ClickHouse را فراهم می‌کند.
    """

    def __init__(self, clickhouse_adapter: ClickHouseAdapter):
        """
        مقداردهی اولیه کلاس فشرده‌سازی داده‌ها

        Args:
            clickhouse_adapter (ClickHouseAdapter): آداپتور اتصال به ClickHouse
        """
        self.clickhouse_adapter = clickhouse_adapter

        # تنظیمات مدیریت داده از کانفیگ مرکزی
        self.data_management_config = config.get_data_management_config()
        logger.info("Data Compressor initialized")

    async def optimize_table(self, table_name: str,
                             final: bool = True,
                             partition: Optional[str] = None,
                             deduplicate: bool = False) -> bool:
        """
        بهینه‌سازی و فشرده‌سازی داده‌های یک جدول در ClickHouse

        این متد فضای ذخیره‌سازی و عملکرد را با فشرده‌سازی داده‌ها و بهینه‌سازی ایندکس‌ها بهبود می‌بخشد.

        Args:
            table_name (str): نام جدول موردنظر برای فشرده‌سازی
            final (bool): اگر True باشد، تمام مراحل بهینه‌سازی انجام می‌شود
            partition (str, optional): نام پارتیشن خاص برای بهینه‌سازی
            deduplicate (bool): اگر True باشد، داده‌های تکراری حذف می‌شوند

        Returns:
            bool: True در صورت موفقیت

        Raises:
            DataManagementError: در صورت بروز خطا در بهینه‌سازی جدول
        """
        # ساخت کوئری با پارامترها برای جلوگیری از SQL Injection
        query_parts = ["OPTIMIZE TABLE"]

        # اضافه کردن نام جدول
        query_parts.append("{table}")

        # اضافه کردن پارامترهای اختیاری
        if partition:
            query_parts.append("PARTITION {partition}")
        if final:
            query_parts.append("FINAL")
        if deduplicate:
            query_parts.append("DEDUPLICATE")

        # ترکیب همه بخش‌های کوئری
        query = " ".join(query_parts)

        # ایجاد پارامترهای کوئری
        params = {
            "table": table_name
        }
        if partition:
            params["partition"] = partition

        try:
            # اجرای کوئری بهینه‌سازی
            await self.clickhouse_adapter.execute(query, params)
            logger.info(f"Table {table_name}{' partition ' + partition if partition else ''} optimized successfully")
            return True

        except Exception as e:
            # ایجاد خطای سفارشی
            error_msg = f"Failed to optimize table {table_name}: {str(e)}"
            logger.error(error_msg)

            raise DataManagementError(
                message=error_msg,
                code="CHE511",
                details={
                    "table_name": table_name,
                    "operation": "optimize",
                    "partition": partition,
                    "error": str(e)
                }
            )

    async def compress_part(self, table_name: str, part_name: str) -> bool:
        """
        فشرده‌سازی یک بخش خاص از داده‌ها

        Args:
            table_name (str): نام جدول
            part_name (str): نام بخش داده

        Returns:
            bool: True در صورت موفقیت

        Raises:
            DataManagementError: در صورت بروز خطا در فشرده‌سازی بخش
        """
        query = "ALTER TABLE {table} COMPRESS PART {part}"
        params = {
            "table": table_name,
            "part": part_name
        }

        try:
            await self.clickhouse_adapter.execute(query, params)
            logger.info(f"Part {part_name} of table {table_name} compressed successfully")
            return True

        except Exception as e:
            error_msg = f"Failed to compress part {part_name} of table {table_name}: {str(e)}"
            logger.error(error_msg)

            raise DataManagementError(
                message=error_msg,
                code="CHE512",
                details={
                    "table_name": table_name,
                    "part_name": part_name,
                    "operation": "compress_part",
                    "error": str(e)
                }
            )

    async def get_storage_stats(self, table_name: str) -> Dict[str, Any]:
        """
        دریافت آمار ذخیره‌سازی یک جدول

        Args:
            table_name (str): نام جدول

        Returns:
            Dict[str, Any]: آمار ذخیره‌سازی جدول

        Raises:
            QueryError: در صورت بروز خطا در اجرای کوئری
        """
        # کوئری برای دریافت آمار ذخیره‌سازی
        query = """
            SELECT 
                table,
                formatReadableSize(sum(bytes)) as size,
                sum(rows) as total_rows,
                max(modification_time) as latest_modification,
                sum(bytes) as bytes_raw,
                count() as parts_count
            FROM system.parts
            WHERE active = 1 AND table = {table} AND database = currentDatabase()
            GROUP BY table
        """

        try:
            result = await self.clickhouse_adapter.execute(query, {"table": table_name})

            if not result or len(result) == 0:
                # اگر نتیجه‌ای یافت نشد، مقادیر پیش‌فرض برگردانده می‌شود
                return {
                    "table": table_name,
                    "size": "0 B",
                    "total_rows": 0,
                    "latest_modification": None,
                    "bytes_raw": 0,
                    "parts_count": 0
                }

            return result[0]

        except Exception as e:
            error_msg = f"Failed to get storage stats for table {table_name}: {str(e)}"
            logger.error(error_msg)

            raise QueryError(
                message=error_msg,
                code="CHE513",
                details={
                    "table_name": table_name,
                    "error": str(e)
                }
            )

    async def optimize_all_tables(self, exclude_tables: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        بهینه‌سازی تمام جداول در دیتابیس فعلی

        Args:
            exclude_tables (List[str], optional): لیست جداولی که نباید بهینه‌سازی شوند

        Returns:
            Dict[str, bool]: وضعیت بهینه‌سازی هر جدول

        Raises:
            OperationalError: در صورت بروز خطا در بهینه‌سازی
        """
        exclude_tables = exclude_tables or []

        # کوئری برای دریافت لیست جداول
        query = "SELECT name FROM system.tables WHERE database = currentDatabase()"

        try:
            # دریافت لیست جداول
            tables_result = await self.clickhouse_adapter.execute(query)
            tables = [row.get('name', '') for row in tables_result if row.get('name') not in exclude_tables]

            # بهینه‌سازی هر جدول
            results = {}
            for table in tables:
                try:
                    results[table] = await self.optimize_table(table)
                except Exception as e:
                    logger.warning(f"Failed to optimize table {table}: {str(e)}")
                    results[table] = False

            return results

        except Exception as e:
            error_msg = f"Failed to optimize all tables: {str(e)}"
            logger.error(error_msg)

            raise OperationalError(
                message=error_msg,
                code="CHE514",
                details={
                    "exclude_tables": exclude_tables,
                    "error": str(e)
                }
            )