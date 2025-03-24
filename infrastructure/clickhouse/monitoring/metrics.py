# infrastructure/clickhouse/monitoring/metrics.py

import os
import logging
from prometheus_client import Gauge, Counter, Histogram, start_http_server

logger = logging.getLogger(__name__)


# تعریف متریک‌های Prometheus
class ClickHouseMetrics:
    """
    کلاس Singleton برای مدیریت متمرکز متریک‌های Prometheus مربوط به ClickHouse
    """
    _instance = None
    _server_started = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ClickHouseMetrics, cls).__new__(cls)
            cls._instance._initialize_metrics()
        return cls._instance

    def _initialize_metrics(self):
        """
        تعریف تمام متریک‌های ClickHouse
        """
        # متریک زمان اجرای کوئری
        self.query_time = Gauge(
            "clickhouse_query_time",
            "Execution time of ClickHouse queries"
        )

        # متریک تعداد اتصالات فعال
        self.active_connections = Gauge(
            "clickhouse_active_connections",
            "Number of active ClickHouse connections"
        )

        # متریک استفاده از دیسک
        self.disk_usage = Gauge(
            "clickhouse_disk_usage",
            "Disk usage in bytes"
        )

        # متریک تعداد کل کوئری‌ها
        self.query_count = Counter(
            "clickhouse_query_count",
            "Total number of executed queries"
        )

        # متریک تعداد خطاها
        self.error_count = Counter(
            "clickhouse_error_count",
            "Total number of query errors"
        )

        # هیستوگرام زمان اجرای کوئری‌ها
        self.query_duration = Histogram(
            "clickhouse_query_duration_seconds",
            "Histogram of query execution times",
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
        )

        logger.info("ClickHouse metrics initialized")

    def start_http_server(self, port=None):
        """
        راه‌اندازی سرور HTTP برای Prometheus

        Args:
            port (int, optional): پورت سرور. اگر None باشد، از متغیر محیطی خوانده می‌شود.
        """
        if ClickHouseMetrics._server_started:
            logger.warning("Prometheus HTTP server is already running")
            return

        port = port or int(os.getenv("PROMETHEUS_PORT", 8000))
        try:
            start_http_server(port)
            ClickHouseMetrics._server_started = True
            logger.info(f"Prometheus metrics server started at http://localhost:{port}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus HTTP server: {str(e)}")

    def record_query_time(self, time_seconds):
        """
        ثبت زمان اجرای کوئری

        Args:
            time_seconds (float): زمان اجرا به ثانیه
        """
        self.query_time.set(time_seconds)
        self.query_duration.observe(time_seconds)
        self.query_count.inc()

    def record_error(self):
        """
        ثبت خطای کوئری
        """
        self.error_count.inc()

    def update_active_connections(self, count):
        """
        بروزرسانی تعداد اتصالات فعال

        Args:
            count (int): تعداد اتصالات فعال
        """
        self.active_connections.set(count)

    def update_disk_usage(self, usage_bytes):
        """
        بروزرسانی میزان استفاده از دیسک

        Args:
            usage_bytes (int): میزان استفاده از دیسک به بایت
        """
        self.disk_usage.set(usage_bytes)


# نمونه‌سازی singleton
metrics = ClickHouseMetrics()
