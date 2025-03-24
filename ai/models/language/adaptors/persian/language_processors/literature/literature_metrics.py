# persian/language_processors/literature/literature_metrics.py
"""
ماژول literature_metrics.py

این ماژول وظیفه جمع‌آوری، تجمیع و گزارش متریک‌های مرتبط با پردازش متون ادبی را بر عهده دارد.
متریک‌هایی مانند تعداد متون ادبی پردازش شده، تعداد درخواست‌ها، تعداد دفعات استفاده از کش،
و سایر آمارهای عملکردی ثبت و گزارش می‌شوند.
ساختار این ماژول مشابه با GrammarMetrics و ProverbMetrics طراحی شده است.
"""

import logging
import time
import json
from typing import Dict, Any, Optional

# واردات کلاس‌های سیستم مانیتورینگ
from ai.core.monitoring.metrics.collector import MetricsCollector
from ai.core.monitoring.metrics.aggregator import MetricsAggregator
from ai.core.monitoring.metrics.exporter import MetricsExporter
from ai.core.monitoring.health.checker import HealthChecker
from ai.core.monitoring.health.reporter import HealthReporter
from ai.core.monitoring.visualization.dashboard_generator import DashboardGenerator
from ai.core.monitoring.visualization.alert_visualizer import AlertVisualizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LiteratureMetrics:
    def __init__(self,
                 collector: Optional[MetricsCollector] = None,
                 aggregator: Optional[MetricsAggregator] = None,
                 exporter: Optional[MetricsExporter] = None,
                 health_checker: Optional[HealthChecker] = None,
                 health_reporter: Optional[HealthReporter] = None,
                 dashboard_generator: Optional[DashboardGenerator] = None,
                 alert_visualizer: Optional[AlertVisualizer] = None):
        """
        سازنده LiteratureMetrics

        Args:
            collector (Optional[MetricsCollector]): شیء جمع‌آوری متریک.
            aggregator (Optional[MetricsAggregator]): شیء تجمیع متریک‌ها.
            exporter (Optional[MetricsExporter]): شیء صادرکننده متریک‌ها.
            health_checker (Optional[HealthChecker]): شیء بررسی سلامت سیستم.
            health_reporter (Optional[HealthReporter]): شیء گزارش‌دهی سلامت.
            dashboard_generator (Optional[DashboardGenerator]): شیء ساخت داشبورد.
            alert_visualizer (Optional[AlertVisualizer]): شیء نمایش هشدارهای مانیتورینگ.
        """
        self.logger = logger
        self.collector = collector
        self.aggregator = aggregator
        self.exporter = exporter
        self.health_checker = health_checker
        self.health_reporter = health_reporter
        self.dashboard_generator = dashboard_generator
        self.alert_visualizer = alert_visualizer

        # دیکشنری داخلی برای ذخیره متریک‌های پردازش متون ادبی
        self.internal_metrics = {
            "requests": 0,
            "cache_hits": 0,
            "literary_text_count": 0,
            "detection_count": 0,
            "processing_time_total": 0.0
        }
        self.logger.info("LiteratureMetrics با موفقیت مقداردهی اولیه شد.")

    def collect_literary_metrics(self,
                                 text_length: int,
                                 literary_text_count: int,
                                 processing_time: float) -> None:
        """
        جمع‌آوری متریک‌های یک نوبت پردازش متون ادبی.

        Args:
            text_length (int): طول متن ورودی.
            literary_text_count (int): تعداد متون ادبی شناسایی شده.
            processing_time (float): زمان پردازش (به ثانیه).
        """
        self.internal_metrics["requests"] += 1
        self.internal_metrics["literary_text_count"] += literary_text_count
        self.internal_metrics["processing_time_total"] += processing_time

        self.logger.debug(f"LiteratureMetrics: text_length={text_length}, "
                          f"literary_text_count={literary_text_count}, "
                          f"processing_time={processing_time:.3f}")

        if self.collector:
            metric_data = {
                "literary_analysis": {
                    "text_length": text_length,
                    "literary_text_count": literary_text_count,
                    "processing_time": processing_time,
                    "timestamp": time.time()
                }
            }
            try:
                self.collector.collect_metrics()
            except Exception as e:
                self.logger.error(f"LiteratureMetrics: خطا در ارسال متریک به Collector: {e}")

    def aggregate_and_export(self) -> None:
        """
        تجمیع متریک‌ها و خروجی گرفتن با استفاده از Aggregator و Exporter.
        """
        if not self.aggregator:
            self.logger.warning("LiteratureMetrics: aggregator تنظیم نشده است.")
            return

        if not self.exporter:
            self.logger.warning("LiteratureMetrics: exporter تنظیم نشده است.")
            return

        try:
            self.aggregator.aggregate_metrics()
        except Exception as e:
            self.logger.error(f"LiteratureMetrics: خطا در تجمیع متریک‌ها: {e}")
            return

        try:
            self.exporter.export_metrics()
            self.logger.info("LiteratureMetrics: متریک‌ها با موفقیت تجمیع و خروجی گرفته شدند.")
        except Exception as e:
            self.logger.error(f"LiteratureMetrics: خطا در خروجی گرفتن متریک‌ها: {e}")

    def generate_dashboard(self) -> None:
        """
        تولید داشبورد متریک‌های پردازش متون ادبی با استفاده از DashboardGenerator.
        """
        if not self.dashboard_generator:
            self.logger.warning("LiteratureMetrics: dashboard_generator تنظیم نشده است.")
            return

        snapshot = self.get_metrics_snapshot()
        try:
            self.dashboard_generator.generate_dashboard()
            self.logger.info("LiteratureMetrics: داشبورد تولید شد.")
        except Exception as e:
            self.logger.error(f"LiteratureMetrics: خطا در تولید داشبورد: {e}")

    def visualize_alerts(self) -> None:
        """
        نمایش هشدارهای مربوط به متریک‌های پردازش متون ادبی با استفاده از AlertVisualizer.
        """
        if not self.alert_visualizer:
            self.logger.warning("LiteratureMetrics: alert_visualizer تنظیم نشده است.")
            return

        try:
            self.alert_visualizer.visualize_alerts()
            self.logger.info("LiteratureMetrics: هشدارها نمایش داده شدند.")
        except Exception as e:
            self.logger.error(f"LiteratureMetrics: خطا در نمایش هشدارها: {e}")

    def check_system_health(self) -> Dict[str, Any]:
        """
        بررسی سلامت سیستم متریک‌ها با استفاده از HealthChecker و HealthReporter.

        Returns:
            dict: گزارشی از وضعیت سلامت.
        """
        if not self.health_checker or not self.health_reporter:
            self.logger.warning("LiteratureMetrics: health_checker یا health_reporter تنظیم نشده‌اند.")
            return {"status": "unavailable", "details": "No health checker or reporter configured."}
        try:
            self.health_checker.run_health_checks()
            self.health_reporter.report_health()
            return {"status": "ok", "details": "Health checks performed and reported."}
        except Exception as e:
            self.logger.error(f"LiteratureMetrics: خطا در بررسی سلامت: {e}")
            return {"status": "error", "details": str(e)}

    def get_metrics_snapshot(self) -> Dict[str, Any]:
        """
        گرفتن نمای کلی از متریک‌های جمع‌آوری‌شده.

        Returns:
            dict: دیکشنری شامل متریک‌های داخلی.
        """
        return dict(self.internal_metrics)

    def reset_metrics(self) -> None:
        """
        بازنشانی متریک‌های داخلی به مقادیر اولیه.
        """
        self.internal_metrics = {
            "requests": 0,
            "cache_hits": 0,
            "literary_text_count": 0,
            "detection_count": 0,
            "processing_time_total": 0.0
        }
        self.logger.info("LiteratureMetrics: متریک‌های داخلی بازنشانی شدند.")


if __name__ == "__main__":
    pm = LiteratureMetrics()
    pm.collect_literary_metrics(text_length=200, literary_text_count=4, processing_time=0.256)
    pm.aggregate_and_export()
    health = pm.check_system_health()
    print("Metrics snapshot:", json.dumps(pm.get_metrics_snapshot(), ensure_ascii=False, indent=2))
    print("Health report:", json.dumps(health, ensure_ascii=False, indent=2))
    pm.reset_metrics()
    print("Metrics after reset:", json.dumps(pm.get_metrics_snapshot(), ensure_ascii=False, indent=2))
