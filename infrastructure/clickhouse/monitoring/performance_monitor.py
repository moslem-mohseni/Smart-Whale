# infrastructure/clickhouse/monitoring/performance_monitor.py

import logging
import time
import asyncio
from ..adapters.connection_pool import ClickHouseConnectionPool
from .metrics import metrics

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    سیستم مانیتورینگ عملکرد ClickHouse و ثبت متریک‌ها
    """

    def __init__(self, port: int = None):
        """
        مقداردهی اولیه سیستم مانیتورینگ

        Args:
            port (int, optional): پورت سرور Prometheus. اگر مقدار None باشد از محیط خوانده می‌شود.
        """
        self.connection_pool = ClickHouseConnectionPool()
        self._running = False

        # راه‌اندازی سرور HTTP Prometheus (اگر هنوز راه‌اندازی نشده باشد)
        metrics.start_http_server(port)

    async def collect_metrics_async(self):
        """
        جمع‌آوری و بروزرسانی متریک‌های سیستم به صورت ناهمگام
        """
        try:
            # بروزرسانی تعداد اتصالات فعال
            active_connections = len(self.connection_pool._pool)
            metrics.update_active_connections(active_connections)

            # اجرای کوئری برای دریافت اطلاعات استفاده از دیسک
            try:
                connection = self.connection_pool.get_connection()
                start_time = time.time()

                result = connection.execute("""
                    SELECT sum(bytes_on_disk) as disk_usage 
                    FROM system.parts
                """)

                query_time = time.time() - start_time
                metrics.record_query_time(query_time)

                if result and len(result) > 0:
                    disk_usage = result[0]['disk_usage'] or 0
                    metrics.update_disk_usage(disk_usage)

                self.connection_pool.release_connection(connection)
            except Exception as e:
                logger.error(f"Error querying system metrics: {str(e)}")
                metrics.record_error()

            logger.debug("Performance metrics collected successfully")
        except Exception as e:
            logger.error(f"Error collecting metrics: {str(e)}")

    def collect_metrics(self):
        """
        جمع‌آوری و بروزرسانی متریک‌های سیستم (نسخه همگام)
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        loop.run_until_complete(self.collect_metrics_async())

    async def start_monitoring_async(self, interval: int = 5):
        """
        اجرای حلقه مانیتورینگ ناهمگام برای بروزرسانی متریک‌ها

        Args:
            interval (int): فاصله زمانی بروزرسانی متریک‌ها (ثانیه)
        """
        self._running = True
        logger.info(f"Starting performance monitoring with interval of {interval} seconds")

        while self._running:
            await self.collect_metrics_async()
            await asyncio.sleep(interval)

    def start_monitoring(self, interval: int = 5):
        """
        اجرای حلقه مانیتورینگ برای بروزرسانی متریک‌ها

        Args:
            interval (int): فاصله زمانی بروزرسانی متریک‌ها (ثانیه)
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(self.start_monitoring_async(interval))
        except KeyboardInterrupt:
            self.stop_monitoring()
            logger.info("Performance monitoring stopped by user")

    def stop_monitoring(self):
        """
        توقف حلقه مانیتورینگ
        """
        self._running = False
        logger.info("Performance monitoring stopped")
