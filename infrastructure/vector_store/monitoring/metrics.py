# infrastructure/vector_store/monitoring/metrics.py

import os
import logging
from prometheus_client import Gauge, Counter, Histogram, CollectorRegistry, start_http_server

logger = logging.getLogger(__name__)


class VectorStoreMetrics:
    """
    کلاس Singleton برای مدیریت متمرکز متریک‌های Prometheus مربوط به Vector Store
    """
    _instance = None
    _server_started = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorStoreMetrics, cls).__new__(cls)
            cls._instance._initialize_metrics()
        return cls._instance

    def _initialize_metrics(self):
        """
        تعریف تمام متریک‌های Vector Store
        """
        # ایجاد رجیستری مستقل
        self.registry = CollectorRegistry()

        # وضعیت اتصال
        self.connection_status = Gauge(
            "vector_store_status",
            "Status of Vector Store connection (1=Online, 0=Offline)",
            registry=self.registry
        )

        # متریک‌های عملیات اصلی
        self.insert_latency = Histogram(
            "vector_store_insert_latency_seconds",
            "Latency of vector insert operations",
            registry=self.registry
        )

        self.delete_latency = Histogram(
            "vector_store_delete_latency_seconds",
            "Latency of vector delete operations",
            registry=self.registry
        )

        self.search_latency = Histogram(
            "vector_store_search_latency_seconds",
            "Latency of vector search operations",
            registry=self.registry
        )

        # شمارنده‌ها
        self.vectors_inserted = Counter(
            "vector_store_vectors_inserted",
            "Total number of vectors inserted",
            registry=self.registry
        )

        self.search_requests = Counter(
            "vector_store_search_requests",
            "Total number of search requests",
            registry=self.registry
        )

        logger.info("Vector Store metrics initialized")

    def start_http_server(self, port=None):
        """
        راه‌اندازی سرور HTTP برای Prometheus

        Args:
            port (int, optional): پورت سرور. اگر None باشد، از متغیر محیطی خوانده می‌شود.
        """
        if VectorStoreMetrics._server_started:
            logger.warning("Prometheus HTTP server is already running")
            return

        port = port or int(os.getenv("PROMETHEUS_PORT", 8000))
        try:
            start_http_server(port, registry=self.registry)
            VectorStoreMetrics._server_started = True
            logger.info(f"Vector Store metrics server started at http://localhost:{port}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus HTTP server: {str(e)}")

    def record_insert_latency(self, time_seconds):
        """ثبت زمان درج"""
        self.insert_latency.observe(time_seconds)
        self.vectors_inserted.inc()

    def record_delete_latency(self, time_seconds):
        """ثبت زمان حذف"""
        self.delete_latency.observe(time_seconds)

    def record_search_latency(self, time_seconds):
        """ثبت زمان جستجو"""
        self.search_latency.observe(time_seconds)
        self.search_requests.inc()

    def set_connection_status(self, is_connected):
        """تنظیم وضعیت اتصال"""
        self.connection_status.set(1 if is_connected else 0)


# نمونه‌سازی singleton برای استفاده در سایر ماژول‌ها
metrics = VectorStoreMetrics()
