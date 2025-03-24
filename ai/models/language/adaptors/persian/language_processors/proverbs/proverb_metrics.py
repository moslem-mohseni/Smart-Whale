# persian/language_processors/proverbs/proverb_metrics.py
"""
ماژول proverb_metrics.py

این ماژول وظیفه جمع‌آوری، تجمیع و گزارش متریک‌های مرتبط با ضرب‌المثل را بر عهده دارد.
متریک‌هایی مانند تعداد ضرب‌المثل‌های پردازش شده، تعداد درخواست‌ها، کش‌ها، نسخه‌های کشف‌شده و سایر متریک‌های عملکردی.
ساختار این ماژول مشابه با GrammarMetrics طراحی شده است.
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


class ProverbMetrics:
    """
    کلاس ProverbMetrics وظیفه دارد متریک‌های مرتبط با پردازش ضرب‌المثل را جمع‌آوری، تجمیع و گزارش دهد.

    ویژگی‌ها:
      - collector: شیء جمع‌آوری متریک از سیستم مانیتورینگ.
      - aggregator: شیء تجمیع متریک‌ها.
      - exporter: شیء صادرکننده متریک‌ها.
      - health_checker: شیء بررسی سلامت سیستم.
      - health_reporter: شیء گزارش‌دهی سلامت.
      - dashboard_generator: شیء تولید داشبورد.
      - alert_visualizer: شیء نمایش هشدارهای مانیتورینگ.
      - internal_metrics: دیکشنری برای نگهداری موقت متریک‌ها.
    """

    def __init__(self,
                 collector: Optional[MetricsCollector] = None,
                 aggregator: Optional[MetricsAggregator] = None,
                 exporter: Optional[MetricsExporter] = None,
                 health_checker: Optional[HealthChecker] = None,
                 health_reporter: Optional[HealthReporter] = None,
                 dashboard_generator: Optional[DashboardGenerator] = None,
                 alert_visualizer: Optional[AlertVisualizer] = None):
        self.logger = logger
        self.collector = collector
        self.aggregator = aggregator
        self.exporter = exporter
        self.health_checker = health_checker
        self.health_reporter = health_reporter
        self.dashboard_generator = dashboard_generator
        self.alert_visualizer = alert_visualizer

        # دیکشنری داخلی برای ذخیره‌سازی متریک‌ها
        self.internal_metrics = {
            "proverb_count": 0,
            "detection_count": 0,
            "new_proverbs": 0,
            "new_variants": 0,
            "cache_hits": 0,
            "requests": 0,
            "total_processing_time": 0.0
        }
        self.logger.info("ProverbMetrics با موفقیت مقداردهی اولیه شد.")

    def collect_proverb_metrics(self,
                                text_length: int,
                                proverb_count: int,
                                processing_time: float) -> None:
        """
        جمع‌آوری متریک‌های یک نوبت پردازش ضرب‌المثل.

        Args:
            text_length (int): طول متن ورودی.
            proverb_count (int): تعداد ضرب‌المثل‌های شناسایی شده.
            processing_time (float): زمان پردازش (ثانیه).
        """
        self.internal_metrics["requests"] += 1
        self.internal_metrics["proverb_count"] += proverb_count
        self.internal_metrics["total_processing_time"] += processing_time

        self.logger.debug(
            f"ProverbMetrics: text_length={text_length}, proverb_count={proverb_count}, processing_time={processing_time:.3f}")

        if self.collector:
            metric_data = {
                "proverb_analysis": {
                    "text_length": text_length,
                    "proverb_count": proverb_count,
                    "processing_time": processing_time,
                    "timestamp": time.time()
                }
            }
            try:
                self.collector.collect_metrics()
            except Exception as e:
                self.logger.error(f"ProverbMetrics: خطا در ارسال متریک به Collector: {e}")

    def aggregate_and_export(self) -> None:
        """
        تجمیع متریک‌ها و خروجی گرفتن با استفاده از Aggregator و Exporter.
        """
        if not self.aggregator:
            self.logger.warning("ProverbMetrics: aggregator تنظیم نشده است.")
            return

        if not self.exporter:
            self.logger.warning("ProverbMetrics: exporter تنظیم نشده است.")
            return

        try:
            self.aggregator.aggregate_metrics()
        except Exception as e:
            self.logger.error(f"ProverbMetrics: خطا در تجمیع متریک‌ها: {e}")
            return

        try:
            self.exporter.export_metrics()
            self.logger.info("ProverbMetrics: متریک‌ها با موفقیت تجمیع و خروجی گرفته شدند.")
        except Exception as e:
            self.logger.error(f"ProverbMetrics: خطا در خروجی گرفتن متریک‌ها: {e}")

    def generate_dashboard(self) -> None:
        """
        تولید داشبورد متریک‌های ضرب‌المثل با استفاده از DashboardGenerator.
        """
        if not self.dashboard_generator:
            self.logger.warning("ProverbMetrics: dashboard_generator تنظیم نشده است.")
            return

        snapshot = self.get_metrics_snapshot()
        try:
            self.dashboard_generator.generate_dashboard()
            self.logger.info("ProverbMetrics: داشبورد تولید شد.")
        except Exception as e:
            self.logger.error(f"ProverbMetrics: خطا در تولید داشبورد: {e}")

    def visualize_alerts(self) -> None:
        """
        نمایش هشدارهای مربوط به متریک‌های ضرب‌المثل با استفاده از AlertVisualizer.
        """
        if not self.alert_visualizer:
            self.logger.warning("ProverbMetrics: alert_visualizer تنظیم نشده است.")
            return

        try:
            self.alert_visualizer.visualize_alerts()
            self.logger.info("ProverbMetrics: هشدارها نمایش داده شدند.")
        except Exception as e:
            self.logger.error(f"ProverbMetrics: خطا در نمایش هشدارها: {e}")

    def check_system_health(self) -> Dict[str, Any]:
        """
        بررسی سلامت سیستم متریک‌ها با استفاده از HealthChecker و HealthReporter.
        """
        if not self.health_checker or not self.health_reporter:
            self.logger.warning("ProverbMetrics: health_checker یا health_reporter تنظیم نشده‌اند.")
            return {"status": "unavailable", "details": "No health checker or reporter configured."}
        try:
            self.health_checker.run_health_checks()
            self.health_reporter.report_health()
            return {"status": "ok", "details": "Health checks performed and reported."}
        except Exception as e:
            self.logger.error(f"ProverbMetrics: خطا در بررسی سلامت: {e}")
            return {"status": "error", "details": str(e)}

    def get_metrics_snapshot(self) -> Dict[str, Any]:
        """
        گرفتن نمای کلی از متریک‌های جمع‌آوری‌شده.
        """
        return dict(self.internal_metrics)

    def reset_metrics(self) -> None:
        """
        بازنشانی متریک‌های داخلی به مقادیر اولیه.
        """
        self.internal_metrics = {
            "proverb_count": 0,
            "detection_count": 0,
            "new_proverbs": 0,
            "new_variants": 0,
            "cache_hits": 0,
            "requests": 0,
            "total_processing_time": 0.0
        }
        self.logger.info("ProverbMetrics: متریک‌های داخلی بازنشانی شدند.")


if __name__ == "__main__":
    # تست نمونه از ProverbMetrics
    pm = ProverbMetrics()
    pm.collect_proverb_metrics(text_length=150, proverb_count=3, processing_time=0.123)
    pm.aggregate_and_export()
    health = pm.check_system_health()
    print("Metrics snapshot:", json.dumps(pm.get_metrics_snapshot(), ensure_ascii=False, indent=2))
    print("Health report:", json.dumps(health, ensure_ascii=False, indent=2))
    pm.reset_metrics()
    print("Metrics after reset:", json.dumps(pm.get_metrics_snapshot(), ensure_ascii=False, indent=2))
