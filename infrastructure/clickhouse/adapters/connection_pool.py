# infrastructure/clickhouse/adapters/connection_pool.py
import logging
from typing import Optional, List, Dict
from threading import Lock, RLock
from clickhouse_driver import Client
from clickhouse_driver.errors import Error as ClickHouseDriverError
from ..config.config import config
from ..exceptions import (
    ConnectionError, PoolExhaustedError, ConnectionTimeoutError,
    AuthenticationError, OperationalError
)

logger = logging.getLogger(__name__)


class ClickHouseConnectionPool:
    """
    مدیریت اتصال‌ها به ClickHouse با استفاده از Connection Pool
    با پشتیبانی از الگوی Singleton بهبودیافته و امکان به‌روزرسانی کانفیگ
    """
    _instance = None
    _lock = Lock()
    _initialized = False

    def __new__(cls, custom_config=None):
        """
        پیاده‌سازی الگوی Singleton با امکان به‌روزرسانی کانفیگ

        Args:
            custom_config: تنظیمات سفارشی (اختیاری)

        Raises:
            OperationalError: در صورت بروز خطا در ایجاد نمونه
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ClickHouseConnectionPool, cls).__new__(cls)

            # اگر کانفیگ سفارشی داده شده یا هنوز مقداردهی اولیه نشده، باید مقداردهی شود
            if not cls._initialized or custom_config is not None:
                cls._instance._initialize(custom_config)
                cls._initialized = True

        return cls._instance

    def __init__(self, custom_config=None):
        """
        متد __init__ برای تعریف تمام متغیرهای نمونه

        Args:
            custom_config: تنظیمات سفارشی (اختیاری)
        """
        # تعریف همه متغیرهای نمونه در اینجا
        # مقداردهی واقعی در _initialize انجام می‌شود
        self.config = None
        self._pool = []
        self._max_connections = 0
        self._active_connections = 0
        self._stats = {}
        self._pool_lock = RLock()

    def _initialize(self, custom_config=None):
        """
        مقداردهی اولیه Connection Pool

        Args:
            custom_config: تنظیمات سفارشی (اختیاری)

        Raises:
            OperationalError: در صورت بروز خطا در مقداردهی اولیه
        """
        try:
            self.config = custom_config or config
            self._pool = []
            self._max_connections = self.config.max_connections
            self._active_connections = 0  # تعداد اتصالات فعال
            self._stats = {
                'created': 0,
                'reused': 0,
                'released': 0,
                'closed': 0
            }

            logger.info(f"ClickHouse Connection Pool initialized with max_connections={self._max_connections}")
        except Exception as e:
            error_msg = f"Failed to initialize connection pool: {str(e)}"
            logger.error(error_msg)
            raise OperationalError(
                message=error_msg,
                code="POOL001"
            )

    def update_config(self, new_config):
        """
        به‌روزرسانی تنظیمات Connection Pool

        Args:
            new_config: تنظیمات جدید

        Raises:
            OperationalError: در صورت بروز خطا در به‌روزرسانی تنظیمات
        """
        try:
            with self._pool_lock:
                # بستن تمامی اتصالات موجود
                self.close_all()
                # به‌روزرسانی تنظیمات
                self.config = new_config
                self._max_connections = new_config.max_connections
                logger.info(f"Connection Pool config updated, max_connections={self._max_connections}")
        except Exception as e:
            error_msg = f"Failed to update connection pool config: {str(e)}"
            logger.error(error_msg)
            raise OperationalError(
                message=error_msg,
                code="POOL002"
            )

    def get_connection(self) -> Client:
        """
        دریافت یک اتصال از pool یا ایجاد یک اتصال جدید در صورت امکان

        Returns:
            Client: شیء اتصال به ClickHouse

        Raises:
            PoolExhaustedError: در صورتی که pool پر باشد و اتصال جدیدی نتواند ایجاد شود
            ConnectionTimeoutError: در صورت timeout در اتصال
            AuthenticationError: در صورت خطای احراز هویت
            ConnectionError: در صورت سایر خطاهای اتصال
        """
        with self._pool_lock:
            try:
                if len(self._pool) > 0:
                    # استفاده مجدد از اتصال موجود
                    connection = self._pool.pop()
                    self._active_connections += 1
                    self._stats['reused'] += 1
                    logger.debug("Reusing existing connection from pool")
                    return connection
                else:
                    # بررسی محدودیت تعداد اتصالات
                    if self._active_connections >= self._max_connections:
                        error_msg = f"Connection pool exhausted. Maximum limit of {self._max_connections} connections reached."
                        logger.error(error_msg)
                        raise PoolExhaustedError(
                            message=error_msg,
                            code="POOL003",
                            max_connections=self._max_connections
                        )

                    # ایجاد اتصال جدید
                    try:
                        connection = Client(**self.config.get_connection_params())
                        # تست اتصال با یک کوئری ساده
                        connection.execute("SELECT 1")
                        self._active_connections += 1
                        self._stats['created'] += 1
                        logger.debug("Created new connection")
                        return connection
                    except ClickHouseDriverError as e:
                        error_str = str(e).lower()
                        if "timeout" in error_str:
                            raise ConnectionTimeoutError(
                                message=f"Connection timeout: {str(e)}",
                                code="POOL004",
                                host=self.config.host,
                                timeout=self.config.connect_timeout
                            )
                        elif "authentication" in error_str or "password" in error_str or "access" in error_str:
                            raise AuthenticationError(
                                message=f"Authentication failed: {str(e)}",
                                code="POOL005",
                                host=self.config.host,
                                user=self.config.user
                            )
                        else:
                            raise ConnectionError(
                                message=f"Failed to connect to ClickHouse: {str(e)}",
                                code="POOL006",
                                host=self.config.host
                            )
            except (PoolExhaustedError, ConnectionTimeoutError, AuthenticationError, ConnectionError):
                # خطاهای سفارشی را مستقیماً منتقل می‌کنیم
                raise
            except Exception as e:
                error_msg = f"Unexpected error getting connection: {str(e)}"
                logger.error(error_msg)
                raise ConnectionError(
                    message=error_msg,
                    code="POOL007",
                    host=self.config.host
                )

    def release_connection(self, connection: Client):
        """
        بازگرداندن اتصال به pool برای استفاده مجدد

        Args:
            connection (Client): اتصال بازگشتی

        Raises:
            OperationalError: در صورت بروز خطا در آزادسازی اتصال
        """
        with self._pool_lock:
            if connection is None:
                return

            try:
                # تست اتصال قبل از بازگرداندن به pool
                connection.execute("SELECT 1")

                if len(self._pool) < self._max_connections:
                    self._pool.append(connection)
                    self._stats['released'] += 1
                    logger.debug("Connection returned to pool")
                else:
                    connection.disconnect()
                    self._stats['closed'] += 1
                    logger.debug("Connection closed (pool full)")

                self._active_connections = max(0, self._active_connections - 1)
            except Exception as e:
                # اگر اتصال مشکل دارد، آن را می‌بندیم
                try:
                    connection.disconnect()
                except:
                    pass
                self._active_connections = max(0, self._active_connections - 1)
                self._stats['closed'] += 1
                logger.warning(f"Connection released but had error: {str(e)}")

    def close_all(self):
        """
        بستن تمامی اتصال‌های باز

        Raises:
            OperationalError: در صورت بروز خطا در بستن اتصالات
        """
        try:
            with self._pool_lock:
                for conn in self._pool:
                    try:
                        conn.disconnect()
                        self._stats['closed'] += 1
                    except Exception as e:
                        logger.warning(f"Error closing connection: {str(e)}")

                self._pool.clear()
                self._active_connections = 0
                logger.info("All connections closed")
        except Exception as e:
            error_msg = f"Error closing all connections: {str(e)}"
            logger.error(error_msg)
            raise OperationalError(
                message=error_msg,
                code="POOL008"
            )

    def get_stats(self) -> Dict[str, int]:
        """
        دریافت آمار استفاده از Connection Pool

        Returns:
            Dict[str, int]: آمار connection pool
        """
        with self._pool_lock:
            stats = self._stats.copy()
            stats.update({
                'pool_size': len(self._pool),
                'active_connections': self._active_connections,
                'max_connections': self._max_connections
            })
            return stats
