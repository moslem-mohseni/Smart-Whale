# infrastructure/clickhouse/monitoring/health_check.py

import logging
import time
from ..adapters.connection_pool import ClickHouseConnectionPool
from .metrics import metrics

logger = logging.getLogger(__name__)


class HealthCheck:
    """
    بررسی سلامت سیستم ClickHouse و پایگاه داده
    """

    def __init__(self):
        """
        مقداردهی اولیه سیستم بررسی سلامت
        """
        self.connection_pool = ClickHouseConnectionPool()

    def check_database_connection(self) -> bool:
        """
        بررسی وضعیت اتصال به ClickHouse

        Returns:
            bool: True در صورت سالم بودن، False در صورت مشکل داشتن اتصال
        """
        try:
            connection = self.connection_pool.get_connection()
            start_time = time.time()

            connection.execute("SELECT 1")

            # ثبت زمان اجرای کوئری در متریک‌ها
            query_time = time.time() - start_time
            metrics.record_query_time(query_time)

            self.connection_pool.release_connection(connection)
            logger.info("ClickHouse connection is healthy")
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
            metrics.record_error()
            return False

    def check_system_health(self) -> dict:
        """
        بررسی کلی سلامت سیستم و منابع

        Returns:
            dict: اطلاعات مربوط به سلامت سیستم
        """
        is_connected = self.check_database_connection()

        health_status = {
            "database": is_connected,
            "status": "healthy" if is_connected else "unhealthy",
            "timestamp": time.time()
        }

        return health_status

    def check_disk_space(self) -> dict:
        """
        بررسی فضای دیسک موجود در سرور ClickHouse

        Returns:
            dict: اطلاعات مربوط به فضای دیسک
        """
        try:
            connection = self.connection_pool.get_connection()
            result = connection.execute("""
                SELECT 
                    formatReadableSize(sum(bytes)) AS disk_usage,
                    formatReadableSize(sum(bytes_on_disk)) AS disk_usage_on_disk,
                    formatReadableSize(sum(data_compressed_bytes)) AS compressed_size,
                    formatReadableSize(sum(data_uncompressed_bytes)) AS uncompressed_size,
                    round(sum(data_compressed_bytes) / sum(data_uncompressed_bytes), 2) AS compression_ratio
                FROM system.parts
            """)
            self.connection_pool.release_connection(connection)

            if result and len(result) > 0:
                return {
                    "success": True,
                    "data": result[0]
                }
            return {
                "success": False,
                "reason": "No data returned"
            }
        except Exception as e:
            logger.error(f"Error checking disk space: {str(e)}")
            return {
                "success": False,
                "reason": str(e)
            }
