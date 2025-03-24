# infrastructure/clickhouse/monitoring/__init__.py

import logging
import os
import threading
from .performance_monitor import PerformanceMonitor
from .health_check import HealthCheck
from .prometheus_exporter import PrometheusExporter
from .metrics import metrics

logger = logging.getLogger(__name__)

# مقداردهی کلاس‌های مانیتورینگ
prometheus_exporter = PrometheusExporter(port=int(os.getenv("PROMETHEUS_PORT", 8000)))
performance_monitor = PerformanceMonitor()
health_check = HealthCheck()


def start_monitoring(background=True, interval=None):
    """
    شروع نظارت بر عملکرد سیستم

    Args:
        background (bool): اجرای مانیتورینگ در پس‌زمینه
        interval (int, optional): فاصله زمانی بین بررسی‌ها (ثانیه)
    """
    monitoring_interval = interval or int(os.getenv("MONITORING_INTERVAL", 5))

    # راه‌اندازی سرور HTTP برای Prometheus
    metrics.start_http_server()

    if background:
        monitoring_thread = threading.Thread(
            target=performance_monitor.start_monitoring,
            args=(monitoring_interval,),
            daemon=True
        )
        monitoring_thread.start()
        logger.info(f"Performance monitoring started in background thread with interval {monitoring_interval}s")
        return monitoring_thread
    else:
        logger.info(f"Starting performance monitoring with interval {monitoring_interval}s")
        performance_monitor.start_monitoring(interval=monitoring_interval)


__all__ = [
    'metrics',
    'prometheus_exporter',
    'performance_monitor',
    'health_check',
    'start_monitoring',
    'PerformanceMonitor',
    'PrometheusExporter',
    'HealthCheck'
]
