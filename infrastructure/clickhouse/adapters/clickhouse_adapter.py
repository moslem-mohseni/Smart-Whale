# infrastructure/clickhouse/adapters/clickhouse_adapter.py
import logging
import asyncio
from typing import Optional, Any, Dict, List, Union
from clickhouse_driver.errors import Error as ClickHouseDriverError
from ..config.config import config
from ..exceptions import (
    ConnectionError, QueryError, QueryExecutionTimeoutError,
    DataTypeError, OperationalError, CircuitBreakerError, RetryExhaustedError
)
from .connection_pool import ClickHouseConnectionPool
from .circuit_breaker import CircuitBreaker
from .retry_mechanism import RetryHandler
from .load_balancer import ClickHouseLoadBalancer

logger = logging.getLogger(__name__)


class ClickHouseAdapter:
    """
    آداپتور مرکزی برای مدیریت اتصال و اجرای درخواست‌ها در ClickHouse

    این کلاس وظیفه هماهنگی بین سیستم‌های مختلف را دارد:
    - Connection Pool
    - Circuit Breaker
    - Retry Mechanism
    - Load Balancer
    """

    def __init__(self, custom_config=None):
        """
        مقداردهی اولیه آداپتور ClickHouse

        Args:
            custom_config: تنظیمات سفارشی (اختیاری)

        Raises:
            OperationalError: در صورت بروز خطا در مقداردهی اولیه
        """
        try:
            self.config = custom_config or config
            logger.info("Initializing ClickHouse Adapter")

            # ایجاد اجزای مورد نیاز
            self.connection_pool = ClickHouseConnectionPool(self.config)
            self.circuit_breaker = CircuitBreaker()
            self.retry_handler = RetryHandler()
            self.load_balancer = ClickHouseLoadBalancer(self.config)

            # تنظیم متغیرهای وضعیت
            self._is_connected = False
            self._last_error = None

            logger.debug("ClickHouse Adapter initialized successfully")
        except Exception as e:
            error_msg = f"Failed to initialize ClickHouse Adapter: {str(e)}"
            logger.error(error_msg)
            raise OperationalError(
                message=error_msg,
                code="ADP002"
            )

    async def connect(self) -> bool:
        """
        اتصال اولیه به ClickHouse برای تست سلامت

        Returns:
            bool: وضعیت موفقیت اتصال

        Raises:
            ConnectionError: در صورت بروز خطا در اتصال به ClickHouse
        """
        try:
            # تلاش برای ایجاد یک اتصال آزمایشی
            connection = self.load_balancer.get_connection()
            connection.execute("SELECT 1")
            self.connection_pool.release_connection(connection)
            self._is_connected = True
            logger.info("Successfully connected to ClickHouse")
            return True
        except Exception as e:
            self._is_connected = False
            self._last_error = str(e)
            error_msg = f"Failed to connect to ClickHouse: {str(e)}"
            logger.error(error_msg)
            raise ConnectionError(
                message=error_msg,
                code="ADP003"
            )

    async def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        اجرای یک کوئری با مدیریت خطا و پشتیبانی از مکانیزم Circuit Breaker و Retry

        Args:
            query (str): کوئری SQL برای اجرا
            params (Dict[str, Any], optional): پارامترهای کوئری برای جلوگیری از SQL Injection

        Returns:
            Any: نتیجه اجرای کوئری

        Raises:
            QueryError: در صورت بروز خطا در کوئری
            ConnectionError: در صورت بروز خطا در اتصال
            CircuitBreakerError: در صورت فعال بودن Circuit Breaker
            RetryExhaustedError: در صورت اتمام تلاش‌های مجدد
            OperationalError: در صورت بروز سایر خطاها
        """
        connection = None
        server = None

        try:
            # دریافت اتصال از Load Balancer
            connection = self.load_balancer.get_connection()

            # ذخیره اطلاعات سرور برای آزادسازی صحیح
            if hasattr(connection, 'connection') and hasattr(connection.connection, 'host'):
                server = connection.connection.host

            # اجرای کوئری با استفاده از Retry Handler
            if params:
                # استفاده از پارامترهای کوئری برای جلوگیری از SQL Injection
                result = self.retry_handler.execute_with_retry(connection.execute, query, params)
            else:
                result = self.retry_handler.execute_with_retry(connection.execute, query)

            return result

        except ClickHouseDriverError as e:
            self._last_error = str(e)
            error_message = f"ClickHouse query execution failed: {str(e)}"
            logger.error(error_message)

            # تبدیل خطای درایور به خطای سفارشی
            if "syntax error" in str(e).lower():
                raise QueryError(
                    message=error_message,
                    code="ADP004",
                    query=query[:100],
                    details={"params": params if params else {}}
                )
            elif "timeout" in str(e).lower():
                raise QueryExecutionTimeoutError(
                    message=error_message,
                    code="ADP005",
                    query=query[:100],
                    timeout=self.config.read_timeout
                )
            elif "type" in str(e).lower():
                raise DataTypeError(
                    message=error_message,
                    code="ADP006",
                    query=query[:100]
                )
            else:
                raise QueryError(
                    message=error_message,
                    code="ADP007",
                    query=query[:100],
                    details={"params": params if params else {}}
                )

        except CircuitBreakerError as e:
            # این خطا را مستقیماً منتقل می‌کنیم
            raise

        except RetryExhaustedError as e:
            # این خطا را مستقیماً منتقل می‌کنیم
            raise

        except ConnectionError as e:
            # این خطا را مستقیماً منتقل می‌کنیم
            raise

        except Exception as e:
            self._last_error = str(e)
            error_message = f"Unexpected error during query execution: {str(e)}"
            logger.error(error_message)
            raise OperationalError(
                message=error_message,
                code="ADP008",
                details={"query": query[:100], "params": params if params else {}}
            )

        finally:
            # آزادسازی منابع در هر صورت
            if connection:
                try:
                    self.connection_pool.release_connection(connection)
                    if server:
                        self.load_balancer.release_connection(server)
                except Exception as e:
                    logger.warning(f"Error releasing connection: {str(e)}")

    async def execute_many(self, queries: List[str]) -> List[Any]:
        """
        اجرای همزمان چندین کوئری

        Args:
            queries (List[str]): لیستی از کوئری‌های SQL

        Returns:
            List[Any]: لیستی از نتایج به همان ترتیب کوئری‌ها
        """
        results = []
        for query in queries:
            try:
                result = await self.execute(query)
                results.append(result)
            except Exception as e:
                results.append(None)
                logger.error(f"Error executing query in batch: {str(e)}")

        return results

    async def execute_with_params(self, query: str, params: Dict[str, Any]) -> Any:
        """
        اجرای کوئری با پارامترهای نام‌گذاری شده

        Args:
            query (str): کوئری SQL با پارامترهای نام‌گذاری شده
            params (Dict[str, Any]): مقادیر پارامترها

        Returns:
            Any: نتیجه اجرای کوئری
        """
        return await self.execute(query, params)

    async def health_check(self) -> Dict[str, Any]:
        """
        بررسی سلامت اتصال به ClickHouse

        Returns:
            Dict[str, Any]: وضعیت سلامت آداپتور
        """
        try:
            connection = self.load_balancer.get_connection()
            start_time = asyncio.get_event_loop().time()
            connection.execute("SELECT 1")
            end_time = asyncio.get_event_loop().time()
            latency = round((end_time - start_time) * 1000, 2)  # میلی‌ثانیه

            self.connection_pool.release_connection(connection)

            return {
                "status": "healthy",
                "latency_ms": latency,
                "pool_stats": self.connection_pool.get_stats(),
                "last_error": self._last_error
            }
        except Exception as e:
            self._last_error = str(e)
            return {
                "status": "unhealthy",
                "error": str(e),
                "pool_stats": self.connection_pool.get_stats(),
                "last_error": self._last_error
            }

    def close(self):
        """
        بستن تمامی اتصالات و آزادسازی منابع
        """
        try:
            self.connection_pool.close_all()
            self.load_balancer.close_all_connections()
            logger.info("ClickHouse Adapter closed all connections")
        except Exception as e:
            logger.error(f"Error closing connections: {str(e)}")
