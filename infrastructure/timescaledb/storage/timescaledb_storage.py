# infrastructure/timescaledb/storage/timescaledb_storage.py

import asyncpg
import logging
from typing import List, Any, Optional, Dict
from ...interfaces import StorageInterface, ConnectionError, OperationError
from ..config.settings import TimescaleDBConfig

logger = logging.getLogger(__name__)

class TimescaleDBStorage(StorageInterface):
    """
    پیاده‌سازی StorageInterface برای TimescaleDB

    این کلاس امکان کار با TimescaleDB را با استفاده از asyncpg فراهم می‌کند.
    قابلیت‌های خاص TimescaleDB مثل hypertable نیز پشتیبانی می‌شوند.
    """

    def __init__(self, config: TimescaleDBConfig):
        """
        مقداردهی اولیه storage با تنظیمات داده شده

        Args:
            config: تنظیمات اتصال به دیتابیس
        """
        self.config = config
        self._pool = None
        self._transaction = None

    async def connect(self) -> None:
        """
        برقراری اتصال به دیتابیس

        Raises:
            ConnectionError: در صورت بروز خطا در اتصال
        """
        try:
            self._pool = await asyncpg.create_pool(
                dsn=self.config.get_connection_string(),
                min_size=self.config.min_connections,
                max_size=self.config.max_connections,
                command_timeout=self.config.connection_timeout
            )
            # فعال‌سازی TimescaleDB
            async with self._pool.acquire() as conn:
                await conn.execute('CREATE EXTENSION IF NOT EXISTS timescaledb')
            logger.info("Successfully connected to TimescaleDB")
        except Exception as e:
            logger.error(f"Failed to connect to TimescaleDB: {str(e)}")
            raise ConnectionError(f"Could not connect to TimescaleDB: {str(e)}")

    async def disconnect(self) -> None:
        """قطع اتصال از دیتابیس"""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Disconnected from TimescaleDB")

    async def is_connected(self) -> bool:
        """
        بررسی وضعیت اتصال

        Returns:
            True اگر اتصال برقرار باشد
        """
        if not self._pool:
            return False
        try:
            async with self._pool.acquire() as conn:
                await conn.execute('SELECT 1')
            return True
        except Exception:
            return False

    async def execute(self, query: str, params: Optional[List[Any]] = None) -> List[Any]:
        """
        اجرای یک پرس‌وجو

        Args:
            query: پرس‌وجوی SQL
            params: پارامترهای پرس‌وجو (اختیاری)

        Returns:
            نتیجه پرس‌وجو

        Raises:
            OperationError: در صورت بروز خطا در اجرای پرس‌وجو
        """
        try:
            if self._transaction:
                result = await self._transaction.fetch(query, *(params or []))
            else:
                async with self._pool.acquire() as conn:
                    result = await conn.fetch(query, *(params or []))
            return [dict(row) for row in result]
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise OperationError(f"Query execution failed: {str(e)}")

    async def execute_many(self, query: str, params_list: List[List[Any]]) -> None:
        """
        اجرای یک پرس‌وجو با چندین سری پارامتر

        Args:
            query: پرس‌وجوی SQL
            params_list: لیست پارامترها

        Raises:
            OperationError: در صورت بروز خطا در اجرای پرس‌وجو
        """
        try:
            if self._transaction:
                await self._transaction.executemany(query, params_list)
            else:
                async with self._pool.acquire() as conn:
                    await conn.executemany(query, params_list)
        except Exception as e:
            logger.error(f"Error executing batch query: {str(e)}")
            raise OperationError(f"Batch query execution failed: {str(e)}")

    async def begin_transaction(self) -> None:
        """
        شروع یک تراکنش جدید

        Raises:
            OperationError: در صورت بروز خطا در شروع تراکنش
        """
        if self._transaction:
            raise OperationError("Transaction already in progress")
        try:
            conn = await self._pool.acquire()
            self._transaction = conn.transaction()
            await self._transaction.start()
        except Exception as e:
            logger.error(f"Error starting transaction: {str(e)}")
            raise OperationError(f"Could not start transaction: {str(e)}")

    async def commit(self) -> None:
        """
        تایید تراکنش جاری

        Raises:
            OperationError: در صورت بروز خطا در تایید تراکنش
        """
        if not self._transaction:
            raise OperationError("No transaction in progress")
        try:
            await self._transaction.commit()
        except Exception as e:
            logger.error(f"Error committing transaction: {str(e)}")
            raise OperationError(f"Could not commit transaction: {str(e)}")
        finally:
            self._transaction = None
            await self._pool.release(self._transaction.connection)

    async def rollback(self) -> None:
        """
        برگشت تراکنش جاری

        Raises:
            OperationError: در صورت بروز خطا در برگشت تراکنش
        """
        if not self._transaction:
            raise OperationError("No transaction in progress")
        try:
            await self._transaction.rollback()
        except Exception as e:
            logger.error(f"Error rolling back transaction: {str(e)}")
            raise OperationError(f"Could not rollback transaction: {str(e)}")
        finally:
            self._transaction = None
            await self._pool.release(self._transaction.connection)

    async def create_table(self, table_name: str, schema: Dict[str, str]) -> None:
        """
        ایجاد جدول جدید

        Args:
            table_name: نام جدول
            schema: ساختار جدول به صورت دیکشنری {نام_ستون: نوع_داده}

        Raises:
            OperationError: در صورت بروز خطا در ایجاد جدول
        """
        try:
            columns = [f"{name} {dtype}" for name, dtype in schema.items()]
            query = f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    {', '.join(columns)}
                )
            """
            await self.execute(query)
            logger.info(f"Created table: {table_name}")
        except Exception as e:
            logger.error(f"Error creating table {table_name}: {str(e)}")
            raise OperationError(f"Could not create table: {str(e)}")

    async def create_hypertable(self, table_name: str, time_column: str) -> None:
        """
        تبدیل یک جدول به hypertable

        Args:
            table_name: نام جدول
            time_column: نام ستون زمان

        Raises:
            OperationError: در صورت بروز خطا در تبدیل جدول
        """
        try:
            query = f"""
                SELECT create_hypertable(
                    '{table_name}',
                    '{time_column}',
                    if_not_exists => TRUE
                )
            """
            await self.execute(query)
            logger.info(f"Created hypertable for: {table_name}")
        except Exception as e:
            logger.error(f"Error creating hypertable for {table_name}: {str(e)}")
            raise OperationError(f"Could not create hypertable: {str(e)}")