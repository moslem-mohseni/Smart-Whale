import logging
from typing import Any, List, Optional
from ..config.connection_pool import ConnectionPool
from ..config.read_write_split import ReadWriteSplitter

logger = logging.getLogger(__name__)


class TimescaleDBStorage:
    """مدیریت اجرای کوئری‌ها در TimescaleDB"""

    def __init__(self, connection_pool: ConnectionPool):
        """
        مقداردهی اولیه کلاس

        Args:
            connection_pool (ConnectionPool): مدیریت Connection Pool
        """
        self.connection_pool = connection_pool
        self.splitter = ReadWriteSplitter(connection_pool)

    async def execute_query(self, query: str, params: Optional[List[Any]] = None) -> List[Any]:
        """
        اجرای یک کوئری در دیتابیس

        Args:
            query (str): متن کوئری SQL
            params (Optional[List[Any]]): مقادیر موردنیاز برای کوئری (در صورت نیاز)

        Returns:
            List[Any]: نتیجه اجرای کوئری
        """
        try:
            result = await self.splitter.execute_query(query, params)
            return result
        except Exception as e:
            logger.error(f"❌ خطا در اجرای کوئری: {e}")
            raise

    async def execute_many(self, query: str, params_list: List[List[Any]]) -> None:
        """
        اجرای یک کوئری برای چندین مجموعه پارامتر

        Args:
            query (str): متن کوئری SQL
            params_list (List[List[Any]]): لیست پارامترها برای هر اجرا
        """
        connection = await self.connection_pool.get_connection(read_only=False)
        try:
            async with connection.transaction():
                await connection.executemany(query, params_list)
            logger.info("✅ اجرای دسته‌ای کوئری با موفقیت انجام شد.")
        except Exception as e:
            logger.error(f"❌ خطا در اجرای دسته‌ای کوئری: {e}")
            raise
        finally:
            await self.connection_pool.release_connection(connection, read_only=False)

    async def begin_transaction(self) -> Any:
        """
        شروع یک تراکنش جدید

        Returns:
            asyncpg.transaction: شیء تراکنش
        """
        connection = await self.connection_pool.get_connection(read_only=False)
        transaction = connection.transaction()
        await transaction.start()
        return transaction, connection

    async def commit_transaction(self, transaction: Any, connection: Any) -> None:
        """
        تأیید تراکنش و اعمال تغییرات

        Args:
            transaction (asyncpg.transaction): شیء تراکنش
            connection (asyncpg.Connection): اتصال پایگاه داده
        """
        try:
            await transaction.commit()
            logger.info("✅ تراکنش با موفقیت تأیید شد.")
        except Exception as e:
            logger.error(f"❌ خطا در تأیید تراکنش: {e}")
            raise
        finally:
            await self.connection_pool.release_connection(connection, read_only=False)

    async def rollback_transaction(self, transaction: Any, connection: Any) -> None:
        """
        لغو تراکنش و بازگرداندن تغییرات

        Args:
            transaction (asyncpg.transaction): شیء تراکنش
            connection (asyncpg.Connection): اتصال پایگاه داده
        """
        try:
            await transaction.rollback()
            logger.info("🔄 تراکنش لغو شد.")
        except Exception as e:
            logger.error(f"❌ خطا در لغو تراکنش: {e}")
            raise
        finally:
            await self.connection_pool.release_connection(connection, read_only=False)
