# infrastructure/clickhouse/integration/stream_processor.py
"""
پردازش داده‌های استریم و درج در ClickHouse
"""

import logging
import json
from typing import Dict, Any, List, Optional, Union
from ..adapters.clickhouse_adapter import ClickHouseAdapter
from ..exceptions import QueryError, DataManagementError, OperationalError
from ..config.config import config
from ..security.audit_log import AuditLogger

logger = logging.getLogger(__name__)


class StreamProcessor:
    """
    پردازش داده‌های استریم و ارسال به ClickHouse

    این کلاس وظیفه دریافت داده‌های استریم (از منابعی مثل Kafka)،
    پردازش آنها و درج در ClickHouse را بر عهده دارد.
    """

    def __init__(self, clickhouse_adapter: ClickHouseAdapter, audit_logger: Optional[AuditLogger] = None):
        """
        مقداردهی اولیه پردازشگر استریم

        Args:
            clickhouse_adapter (ClickHouseAdapter): آداپتور اتصال به ClickHouse
            audit_logger (AuditLogger, optional): لاگر رخدادهای امنیتی
        """
        self.clickhouse_adapter = clickhouse_adapter
        self.audit_logger = audit_logger

        # ایجاد خودکار لاگر امنیتی اگر ارائه نشده باشد
        if self.audit_logger is None:
            from ..security import create_audit_logger
            self.audit_logger = create_audit_logger(app_name="stream_processor")

        logger.info("Stream Processor initialized")

    async def process_stream_data(self, table_name: str, data: Union[Dict[str, Any], List[Dict[str, Any]]]):
        """
        پردازش داده‌های دریافتی از استریم و درج در ClickHouse با استفاده از پارامترهای کوئری
        برای جلوگیری از SQL Injection

        Args:
            table_name (str): نام جدول هدف برای ذخیره داده‌ها
            data (Dict[str, Any] | List[Dict[str, Any]]): داده‌های استریم که باید درج شوند.
                می‌تواند یک دیکشنری یا لیستی از دیکشنری‌ها باشد.

        Raises:
            DataManagementError: در صورت بروز خطا در پردازش داده‌ها
        """
        try:
            # تبدیل داده‌ی تکی به لیست برای پردازش یکسان
            if isinstance(data, dict):
                data_list = [data]
            else:
                data_list = data

            if not data_list:
                logger.warning("Empty data received for stream processing")
                return

            # پردازش و درج داده‌ها
            await self._insert_data_batch(table_name, data_list)

            # ثبت لاگ موفقیت
            self._log_successful_operation(table_name, len(data_list))

        except (QueryError, OperationalError) as e:
            # خطاهای سفارشی را مستقیماً منتقل می‌کنیم
            logger.error(f"Failed to process stream data: {str(e)}")
            self._log_failed_operation(table_name, data, e)
            raise

        except Exception as e:
            # سایر خطاها را به DataManagementError تبدیل می‌کنیم
            error_msg = f"Error processing stream data for table {table_name}: {str(e)}"
            logger.error(error_msg)
            self._log_failed_operation(table_name, data, e)
            raise DataManagementError(
                message=error_msg,
                code="SPR001",
                operation="stream_insert",
                table_name=table_name
            )

    async def _insert_data_batch(self, table_name: str, data_list: List[Dict[str, Any]]):
        """
        درج دسته‌ای داده‌ها در ClickHouse با استفاده از پارامترهای کوئری

        Args:
            table_name (str): نام جدول هدف
            data_list (List[Dict[str, Any]]): لیست داده‌ها برای درج

        Raises:
            QueryError: در صورت بروز خطا در اجرای کوئری
            OperationalError: در صورت بروز خطای عملیاتی
        """
        # استخراج ستون‌ها از اولین رکورد
        columns = list(data_list[0].keys())

        # ساخت کوئری INSERT با پارامترهای امن
        placeholders = ", ".join([f":{col}" for col in columns])
        columns_str = ", ".join(columns)

        # استفاده از VALUES ساده برای دسته‌های کوچک
        if len(data_list) == 1:
            query = f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})"
            await self.clickhouse_adapter.execute(query, data_list[0])

        # استفاده از INSERT با چندین VALUES برای دسته‌های متوسط
        elif len(data_list) <= 1000:
            values_placeholders = []
            params = {}

            for i, row in enumerate(data_list):
                row_placeholders = []
                for col in columns:
                    param_key = f"{col}_{i}"
                    row_placeholders.append(f":{param_key}")
                    params[param_key] = row.get(col)

                values_placeholders.append(f"({', '.join(row_placeholders)})")

            query = f"INSERT INTO {table_name} ({columns_str}) VALUES {', '.join(values_placeholders)}"
            await self.clickhouse_adapter.execute(query, params)

        # استفاده از JSON یا روش دسته‌ای مناسب برای دسته‌های بزرگ
        else:
            # در ClickHouse، برای درج حجم زیاد داده، روش‌های بهینه‌تری مانند استفاده از
            # فرمت JSON یا استفاده از درج دسته‌ای وجود دارد
            formatted_data = json.dumps(data_list)
            query = f"""
                INSERT INTO {table_name} 
                FORMAT JSONEachRow
                {formatted_data}
            """
            await self.clickhouse_adapter.execute(query)

        logger.info(f"Successfully inserted {len(data_list)} records into {table_name}")

    def _log_successful_operation(self, table_name: str, record_count: int):
        """
        ثبت لاگ برای عملیات موفق

        Args:
            table_name (str): نام جدول
            record_count (int): تعداد رکوردهای پردازش شده
        """
        if self.audit_logger:
            details = {
                "table_name": table_name,
                "record_count": record_count,
                "operation": "stream_insert"
            }
            self.audit_logger.log_event(
                username="stream_processor",
                action="data_insert",
                status="success",
                details=json.dumps(details),
                resource=table_name
            )

    def _log_failed_operation(self, table_name: str, data: Any, error: Exception):
        """
        ثبت لاگ برای عملیات ناموفق

        Args:
            table_name (str): نام جدول
            data (Any): داده‌های مورد پردازش
            error (Exception): خطای رخ داده
        """
        if self.audit_logger:
            if isinstance(data, (list, dict)):
                record_count = len(data) if isinstance(data, list) else 1
            else:
                record_count = 0

            details = {
                "table_name": table_name,
                "record_count": record_count,
                "operation": "stream_insert",
                "error": str(error),
                "error_type": error.__class__.__name__
            }

            self.audit_logger.log_event(
                username="stream_processor",
                action="data_insert",
                status="failure",
                details=json.dumps(details),
                resource=table_name
            )
