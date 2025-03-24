"""
ماژول `monitoring/` وظیفه‌ی مدیریت سلامت و متریک‌های عملکردی سیستم پردازش زبان را بر عهده دارد.

📌 اجزای اصلی این ماژول:
- `health_check.py` → بررسی سلامت سرویس‌های زیرساختی مرتبط با پردازش زبان
- `performance_metrics.py` → جمع‌آوری و پردازش متریک‌های عملکردی
"""

from .health_check import HealthCheck
from .performance_metrics import PerformanceMetrics
from ai.core.monitoring.metrics.collector import MetricsCollector
from ai.core.monitoring.metrics.aggregator import MetricsAggregator
from ai.core.monitoring.metrics.exporter import MetricsExporter
from ai.core.monitoring.health.checker import HealthChecker
from ai.core.monitoring.health.reporter import HealthReporter
from infrastructure.monitoring.health_service import HealthService

# مقداردهی اولیه سرویس‌های مانیتورینگ
health_checker = HealthChecker()
health_reporter = HealthReporter()
metrics_collector = MetricsCollector()
metrics_aggregator = MetricsAggregator()
metrics_exporter = MetricsExporter()
health_service = HealthService()

# مقداردهی اولیه HealthCheck و PerformanceMetrics
health_check = HealthCheck(health_checker, health_reporter, metrics_collector, health_service)
performance_metrics = PerformanceMetrics(metrics_collector, metrics_aggregator, metrics_exporter, health_check)

__all__ = [
    "health_check",
    "performance_metrics",
    "HealthCheck",
    "PerformanceMetrics",
]
