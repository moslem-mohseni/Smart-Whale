# persian/language_processors/grammar/metrics.py
"""
ماژول metrics.py

این ماژول وظیفه مدیریت و جمع‌آوری متریک‌های مربوط به پردازش گرامری زبان فارسی را بر عهده دارد.
از سیستم مانیتورینگ مرکزی شامل Collector، Aggregator، Exporter، و ابزارهای گزارش سلامت و داشبورد استفاده می‌کند.
متریک‌های مختلفی مانند تعداد درخواست‌های تحلیل، تعداد خطاهای شناسایی‌شده، میانگین سطح اطمینان و زمان پردازش جمع‌آوری و گزارش می‌شود.

نمونه استفاده:
    از این ماژول می‌توان در سایر بخش‌های سیستم برای ثبت و گزارش متریک‌های پردازش گرامر استفاده کرد.
    (به مستندات مثال در داکیومنت اشاره شده است)
"""

import logging
import time
import json
from typing import Optional, Dict, Any

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


class GrammarMetrics:
    """
    کلاس GrammarMetrics وظیفه جمع‌آوری و مدیریت متریک‌های تحلیل و اصلاح گرامر فارسی را دارد.

    ویژگی‌ها:
        collector (Optional[MetricsCollector]): شیء جمع‌آوری متریک‌ها از سیستم مانیتورینگ.
        aggregator (Optional[MetricsAggregator]): شیء تجمیع متریک‌ها.
        exporter (Optional[MetricsExporter]): شیء صادرکننده متریک‌ها به سیستم‌های خارجی.
        health_checker (Optional[HealthChecker]): شیء بررسی سلامت سیستم.
        health_reporter (Optional[HealthReporter]): شیء گزارش‌دهی سلامت.
        dashboard_generator (Optional[DashboardGenerator]): شیء تولید داشبورد.
        alert_visualizer (Optional[AlertVisualizer]): شیء نمایش هشدارهای مانیتورینگ.
        internal_metrics (dict): ساختاری برای نگهداری موقت متریک‌های جمع‌آوری‌شده.
    """

    def __init__(self,
                 collector: Optional[MetricsCollector] = None,
                 aggregator: Optional[MetricsAggregator] = None,
                 exporter: Optional[MetricsExporter] = None,
                 health_checker: Optional[HealthChecker] = None,
                 health_reporter: Optional[HealthReporter] = None,
                 dashboard_generator: Optional[DashboardGenerator] = None,
                 alert_visualizer: Optional[AlertVisualizer] = None):
        """
        سازنده GrammarMetrics.

        Args:
            collector (Optional[MetricsCollector]): شیء برای جمع‌آوری متریک‌ها.
            aggregator (Optional[MetricsAggregator]): شیء برای تجمیع متریک‌ها.
            exporter (Optional[MetricsExporter]): شیء برای خروجی گرفتن از متریک‌ها.
            health_checker (Optional[HealthChecker]): شیء بررسی سلامت.
            health_reporter (Optional[HealthReporter]): شیء گزارش‌دهی سلامت.
            dashboard_generator (Optional[DashboardGenerator]): شیء تولید داشبورد.
            alert_visualizer (Optional[AlertVisualizer]): شیء نمایش هشدارها.
        """
        self.logger = logger
        self.collector = collector
        self.aggregator = aggregator
        self.exporter = exporter
        self.health_checker = health_checker
        self.health_reporter = health_reporter
        self.dashboard_generator = dashboard_generator
        self.alert_visualizer = alert_visualizer

        # دیکشنری داخلی برای ذخیره‌سازی متریک‌ها تا زمان تجمیع
        self.internal_metrics = {
            "analysis_count": 0,
            "total_errors": 0,
            "avg_confidence": 0.0,
            "total_processing_time": 0.0
        }
        self.logger.info("GrammarMetrics با موفقیت مقداردهی اولیه شد.")

    def collect_grammar_analysis_metrics(self,
                                         text_length: int,
                                         error_count: int,
                                         confidence: float,
                                         source: str,
                                         processing_time: float = 0.0) -> None:
        """
        جمع‌آوری متریک‌های یک نوبت تحلیل گرامر.

        Args:
            text_length (int): طول متن (تعداد کاراکتر یا کلمات).
            error_count (int): تعداد خطاهای شناسایی‌شده.
            confidence (float): سطح اطمینان تحلیل (0 تا 1).
            source (str): منبع تحلیل (مثلاً "rule_based"، "teacher"، "smart_model").
            processing_time (float): مدت زمان پردازش به ثانیه.
        """
        self.internal_metrics["analysis_count"] += 1
        self.internal_metrics["total_errors"] += error_count
        self.internal_metrics["total_processing_time"] += processing_time

        # به‌روزرسانی میانگین سطح اطمینان
        count = self.internal_metrics["analysis_count"]
        old_avg = self.internal_metrics["avg_confidence"]
        new_avg = old_avg + (confidence - old_avg) / count
        self.internal_metrics["avg_confidence"] = new_avg

        # ارسال متریک‌ها به سیستم مانیتورینگ (در صورت موجود بودن collector)
        if self.collector:
            metric_data = {
                "grammar_analysis": {
                    "text_length": text_length,
                    "error_count": error_count,
                    "confidence": confidence,
                    "source": source,
                    "processing_time": processing_time,
                    "timestamp": time.time()
                }
            }
            try:
                self.collector.collect_metrics()  # در صورت نیاز می‌توان metric_data را نیز ارسال کرد
            except Exception as e:
                self.logger.error(f"[GrammarMetrics] خطا در ارسال متریک به Collector: {e}")

        self.logger.debug(f"[GrammarMetrics] متریک جمع‌آوری شد: text_length={text_length}, "
                          f"error_count={error_count}, confidence={confidence:.2f}, "
                          f"source={source}, processing_time={processing_time:.3f}")

    def aggregate_and_export(self) -> None:
        """
        تجمیع متریک‌ها با استفاده از Aggregator و سپس خروجی گرفتن با استفاده از Exporter.
        """
        if not self.aggregator:
            self.logger.warning("[GrammarMetrics] aggregator تنظیم نشده است. تجمیع متریک‌ها امکان‌پذیر نیست.")
            return

        if not self.exporter:
            self.logger.warning("[GrammarMetrics] exporter تنظیم نشده است. خروجی گرفتن متریک‌ها امکان‌پذیر نیست.")
            return

        try:
            self.aggregator.aggregate_metrics()
        except Exception as e:
            self.logger.error(f"[GrammarMetrics] خطا در فراخوانی aggregator.aggregate_metrics(): {e}")
            return

        try:
            self.exporter.export_metrics()
            self.logger.info("[GrammarMetrics] متریک‌ها با موفقیت تجمیع و خروجی گرفته شدند.")
        except Exception as e:
            self.logger.error(f"[GrammarMetrics] خطا در خروجی گرفتن متریک‌ها: {e}")

    def generate_dashboard(self) -> None:
        """
        تولید داشبورد با استفاده از DashboardGenerator.
        """
        if not self.dashboard_generator:
            self.logger.warning("[GrammarMetrics] dashboard_generator تنظیم نشده است. امکان ساخت داشبورد وجود ندارد.")
            return

        snapshot = self.get_metrics_snapshot()
        try:
            self.dashboard_generator.generate_dashboard()
            self.logger.info("[GrammarMetrics] داشبورد تولید شد.")
        except Exception as e:
            self.logger.error(f"[GrammarMetrics] خطا در ساخت داشبورد: {e}")

    def visualize_alerts(self) -> None:
        """
        نمایش هشدارهای مانیتورینگ با استفاده از AlertVisualizer.
        """
        if not self.alert_visualizer:
            self.logger.warning("[GrammarMetrics] alert_visualizer تنظیم نشده است. امکان نمایش هشدارها وجود ندارد.")
            return

        try:
            self.alert_visualizer.visualize_alerts()
            self.logger.info("[GrammarMetrics] هشدارها نمایش داده شدند.")
        except Exception as e:
            self.logger.error(f"[GrammarMetrics] خطا در نمایش هشدارها: {e}")

    def check_system_health(self) -> Dict[str, Any]:
        """
        بررسی سلامت سیستم با استفاده از HealthChecker و گزارش سلامت توسط HealthReporter.

        Returns:
            dict: گزارش سلامت سیستم شامل وضعیت و جزئیات.
        """
        if not self.health_checker or not self.health_reporter:
            self.logger.warning("[GrammarMetrics] health_checker یا health_reporter تنظیم نشده‌اند.")
            return {"status": "unavailable", "details": "No health checker or reporter configured."}

        try:
            self.health_checker.run_health_checks()
            self.health_reporter.report_health()
            return {"status": "ok", "details": "Health checks performed and reported."}
        except Exception as e:
            self.logger.error(f"[GrammarMetrics] خطا در بررسی سلامت: {e}")
            return {"status": "error", "details": str(e)}

    def get_metrics_snapshot(self) -> Dict[str, Any]:
        """
        گرفتن نمایی از متریک‌های درون‌کلاسی.

        Returns:
            dict: دیکشنری شامل اطلاعات متریک‌های فعلی مانند analysis_count، total_errors، avg_confidence و total_processing_time.
        """
        return dict(self.internal_metrics)

    def reset_metrics(self) -> None:
        """
        بازنشانی متریک‌های درون‌کلاسی به مقادیر اولیه.
        """
        self.internal_metrics = {
            "analysis_count": 0,
            "total_errors": 0,
            "avg_confidence": 0.0,
            "total_processing_time": 0.0
        }
        self.logger.info("[GrammarMetrics] متریک‌های داخلی بازنشانی شدند.")


# نمونه تست مستقل
if __name__ == "__main__":
    # مثال استفاده از GrammarMetrics:
    # (این مثال فرض می‌کند که اشیاء collector، aggregator، exporter، health_checker، health_reporter،
    #  dashboard_generator و alert_visualizer از سیستم مانیتورینگ موجود هستند.)

    # ایجاد نمونه GrammarMetrics با مقادیر پیش‌فرض (در صورت عدم وجود سیستم مانیتورینگ، برخی متدها هشدار می‌دهند)
    grammar_metrics = GrammarMetrics()

    # جمع‌آوری یک متریک نمونه
    grammar_metrics.collect_grammar_analysis_metrics(
        text_length=120,
        error_count=5,
        confidence=0.82,
        source="rule_based",
        processing_time=0.045
    )

    # تجمیع و خروجی گرفتن متریک‌ها
    grammar_metrics.aggregate_and_export()

    # تولید داشبورد
    grammar_metrics.generate_dashboard()

    # بررسی سلامت سیستم
    health_report = grammar_metrics.check_system_health()
    print("گزارش سلامت سیستم:", json.dumps(health_report, ensure_ascii=False, indent=2))

    # نمایش نمای کلی از متریک‌های جمع‌آوری‌شده
    snapshot = grammar_metrics.get_metrics_snapshot()
    print("نمای کلی متریک‌ها:", json.dumps(snapshot, ensure_ascii=False, indent=2))

    # بازنشانی متریک‌ها
    grammar_metrics.reset_metrics()
