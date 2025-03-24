import logging
from typing import Any, Optional, List
from .connection_pool import ConnectionPool

logger = logging.getLogger(__name__)


class ReadWriteSplitter:
    """مدیریت تقسیم بار بین پایگاه داده اصلی و Read Replica"""

    def __init__(self, connection_pool: ConnectionPool):
        """
        مقداردهی اولیه

        Args:
            connection_pool (ConnectionPool): مدیریت Connection Pool
        """
        self.connection_pool = connection_pool

    async def execute_query(self, query: str, params: Optional[List[Any]] = None) -> List[Any]:
        """
        اجرای یک کوئری با تشخیص خودکار Read/Write

        Args:
            query (str): متن کوئری SQL
            params (Optional[List[Any]]): مقادیر موردنیاز برای پارامترهای کوئری (در صورت نیاز)

        Returns:
            List[Any]: نتیجه کوئری
        """
        read_only = query.strip().lower().startswith("select")
        connection = await self.connection_pool.get_connection(read_only=read_only)

        try:
            logger.info(f"🔄 اجرای کوئری {'(READ)' if read_only else '(WRITE)'}: {query}")
            result = await connection.fetch(query, *params) if params else await connection.fetch(query)
            return [dict(row) for row in result]  # تبدیل نتیجه به دیکشنری
        except Exception as e:
            logger.error(f"❌ خطا در اجرای کوئری: {e}")
            raise
        finally:
            await self.connection_pool.release_connection(connection, read_only=read_only)
