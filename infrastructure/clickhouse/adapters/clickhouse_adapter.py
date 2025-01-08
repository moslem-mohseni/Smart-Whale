# infrastructure/clickhouse/adapters/clickhouse_adapter.py

from clickhouse_driver import Client
from clickhouse_driver.errors import Error as ClickHouseError
import logging
from typing import Any, List, Dict, Optional
from datetime import datetime

from ..config.settings import ClickHouseConfig, QuerySettings
from ..domain.models import AnalyticsQuery, AnalyticsResult, TableSchema
from ...interfaces import StorageInterface, ConnectionError, OperationError

logger = logging.getLogger(__name__)


class ClickHouseAdapter(StorageInterface):
    """
    تطبیق‌دهنده برای ارتباط با ClickHouse

    این کلاس مسئولیت برقراری ارتباط با ClickHouse و اجرای عملیات پایه و پیشرفته را بر عهده دارد.
    علاوه بر پیاده‌سازی StorageInterface، قابلیت‌های تحلیلی پیشرفته نیز ارائه می‌دهد.
    """

    def __init__(self, config: ClickHouseConfig):
        """
        راه‌اندازی adapter با تنظیمات داده شده

        Args:
            config: تنظیمات اتصال به ClickHouse
        """
        self.config = config
        self._client = None
        self.query_settings = QuerySettings()
        self._transaction = None

    async def connect(self) -> None:
        """برقراری اتصال به ClickHouse"""
        try:
            # ClickHouse-driver به صورت sync کار می‌کند، اما ما آن را در یک wrapper async قرار می‌دهیم
            self._client = Client(**self.config.get_connection_params())
            # تست اتصال
            self._client.execute('SELECT 1')
            logger.info("Successfully connected to ClickHouse")

        except ClickHouseError as e:
            logger.error(f"Failed to connect to ClickHouse: {str(e)}")
            raise ConnectionError(f"Could not connect to ClickHouse: {str(e)}")

    async def disconnect(self) -> None:
        """قطع اتصال از ClickHouse"""
        if self._client:
            self._client.disconnect()
            self._client = None
            logger.info("Disconnected from ClickHouse")

    async def is_connected(self) -> bool:
        """بررسی وضعیت اتصال"""
        if not self._client:
            return False
        try:
            self._client.execute('SELECT 1')
            return True
        except Exception:
            return False

    async def execute(self, query: str, params: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
        """
        اجرای یک پرس‌وجو

        Args:
            query: پرس‌وجوی SQL
            params: پارامترهای پرس‌وجو (اختیاری)

        Returns:
            نتایج پرس‌وجو به صورت لیست دیکشنری
        """
        if not await self.is_connected():
            raise ConnectionError("Not connected to ClickHouse")

        try:
            settings = self.query_settings.to_dict()
            if self._transaction:
                settings['mutations_sync'] = 1

            result = self._client.execute(
                query,
                params or [],
                with_column_types=True,
                settings=settings
            )

            # تبدیل نتایج به فرمت مناسب
            rows, columns = result
            column_names = [col[0] for col in columns]
            return [dict(zip(column_names, row)) for row in rows]

        except ClickHouseError as e:
            logger.error(f"Query execution failed: {str(e)}")
            raise OperationError(f"Query execution failed: {str(e)}")

    async def execute_many(self, query: str, params_list: List[List[Any]]) -> None:
        """
        اجرای یک پرس‌وجو با چندین سری پارامتر

        Args:
            query: پرس‌وجوی SQL
            params_list: لیست پارامترها
        """
        if not await self.is_connected():
            raise ConnectionError("Not connected to ClickHouse")

        try:
            settings = self.query_settings.to_dict()
            if self._transaction:
                settings['mutations_sync'] = 1

            self._client.execute(
                query,
                params_list,
                settings=settings
            )

        except ClickHouseError as e:
            logger.error(f"Batch query execution failed: {str(e)}")
            raise OperationError(f"Batch query execution failed: {str(e)}")

    async def begin_transaction(self) -> None:
        """شروع یک تراکنش"""
        if self._transaction:
            raise OperationError("Transaction already in progress")
        self._transaction = True

    async def commit(self) -> None:
        """تایید تراکنش"""
        if not self._transaction:
            raise OperationError("No transaction in progress")
        self._transaction = None

    async def rollback(self) -> None:
        """برگشت تراکنش

        Note: ClickHouse از rollback پشتیبانی نمی‌کند، اما برای سازگاری با interface این متد را داریم.
        """
        if not self._transaction:
            raise OperationError("No transaction in progress")
        self._transaction = None

    async def create_table(self, table_name: str, schema: Dict[str, str]) -> None:
        """
        ایجاد جدول جدید

        Args:
            table_name: نام جدول
            schema: ساختار جدول
        """
        columns = [f"{name} {type_}" for name, type_ in schema.items()]
        query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                {', '.join(columns)}
            ) ENGINE = MergeTree()
            ORDER BY tuple()
        """
        await self.execute(query)

    async def create_hypertable(self, table_name: str, time_column: str) -> None:
        """
        تبدیل یک جدول به hypertable

        Note: ClickHouse نیازی به hypertable ندارد چون خودش برای داده‌های زمانی بهینه است.
        این متد برای سازگاری با interface پیاده‌سازی شده است.

        Args:
            table_name: نام جدول
            time_column: نام ستون زمان
        """
        logger.info(f"ClickHouse does not need hypertable conversion for {table_name}")
        return

    async def execute_analytics_query(self, query: AnalyticsQuery) -> List[Dict[str, Any]]:
        """
        اجرای پرس‌وجوی تحلیلی

        Args:
            query: پرس‌وجوی تحلیلی

        Returns:
            نتایج تحلیل
        """
        # تبدیل AnalyticsQuery به SQL
        sql_parts = ["SELECT"]

        # اضافه کردن ستون‌ها
        select_columns = query.dimensions + query.metrics
        sql_parts.append(", ".join(select_columns))

        # اضافه کردن جدول
        sql_parts.append("FROM events")

        # اضافه کردن شرط‌ها
        where_conditions = []
        params = []

        if query.time_range:
            start_time, end_time = query.time_range
            where_conditions.append("timestamp BETWEEN ? AND ?")
            params.extend([start_time, end_time])

        if query.filters:
            for key, value in query.filters.items():
                where_conditions.append(f"{key} = ?")
                params.append(value)

        if where_conditions:
            sql_parts.append("WHERE " + " AND ".join(where_conditions))

        # اضافه کردن گروه‌بندی
        if query.dimensions:
            sql_parts.append("GROUP BY " + ", ".join(query.dimensions))

        # اضافه کردن ترتیب
        if query.order_by:
            sql_parts.append("ORDER BY " + ", ".join(query.order_by))

        # اضافه کردن محدودیت
        if query.limit:
            sql_parts.append(f"LIMIT {query.limit}")

        # اجرای پرس‌وجو
        final_sql = " ".join(sql_parts)
        return await self.execute(final_sql, params)