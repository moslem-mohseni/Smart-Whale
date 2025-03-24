# infrastructure/clickhouse/monitoring/prometheus_exporter.py

import os
import logging
import time
import asyncio
from .metrics import metrics

logger = logging.getLogger(__name__)


class PrometheusExporter:
    """
    ماژول Prometheus برای مانیتورینگ ClickHouse
    """

    def __init__(self, port: int = None):
        """
        مقداردهی اولیه سرور Prometheus

        Args:
            port (int, optional): پورت سرور متریک‌ها. اگر مقدار None باشد، از محیط خوانده می‌شود.
        """
        self._running = False

        # راه‌اندازی سرور HTTP Prometheus (اگر هنوز راه‌اندازی نشده باشد)
        metrics.start_http_server(port)

    def update_metrics(self, query_time: float = None, active_connections: int = None, disk_usage: int = None):
        """
        بروزرسانی متریک‌های سیستم

        Args:
            query_time (float, optional): زمان اجرای کوئری‌ها
            active_connections (int, optional): تعداد اتصالات فعال
            disk_usage (int, optional): میزان استفاده از دیسک (بایت)
        """
        if query_time is not None:
            metrics.record_query_time(query_time)

        if active_connections is not None:
            metrics.update_active_connections(active_connections)

        if disk_usage is not None:
            metrics.update_disk_usage(disk_usage)

    async def start_monitoring_async(self, interval: int = 5):
        """
        اجرای حلقه مانیتورینگ ناهمگام برای بروزرسانی متریک‌ها

        Args:
            interval (int): فاصله زمانی بروزرسانی متریک‌ها (ثانیه)
        """
        self._running = True
        logger.info(f"Starting Prometheus metrics export with interval of {interval} seconds")

        while self._running:
            # در اینجا مقدار متریک‌ها از منبع داده خارجی خوانده می‌شود
            # این یک مثال است - در کاربرد واقعی این مقادیر باید از سرور ClickHouse دریافت شوند
            self.update_metrics(
                query_time=0.01,
                active_connections=5,
                disk_usage=1024000
            )
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
            logger.info("Prometheus metrics export stopped by user")

    def stop_monitoring(self):
        """
        توقف حلقه مانیتورینگ
        """
        self._running = False
        logger.info("Prometheus metrics export stopped")
