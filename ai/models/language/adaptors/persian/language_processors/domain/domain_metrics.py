# persian/language_processors/domain/domain_metrics.py

"""
ماژول domain_metrics.py

این ماژول وظیفه جمع‌آوری، تجمیع و گزارش متریک‌های مرتبط با مدیریت دانش حوزه‌ای (Domain Knowledge) را بر عهده دارد.
از ساختاری مشابه با GrammarMetrics و LiteratureMetrics بهره می‌بریم.
"""

import logging
import time
import json
from typing import Dict, Any, Optional

# فرض بر این است که کلاس‌های زیر در سیستم مانیتورینگ مرکزی تعریف شده‌اند:
# اگر پروژه شما متفاوت است، بسته به ساختار واقعی تغییر دهید.
from ai.core.monitoring.metrics.collector import MetricsCollector
from ai.core.monitoring.metrics.aggregator import MetricsAggregator
from ai.core.monitoring.metrics.exporter import MetricsExporter
from ai.core.monitoring.health.checker import HealthChecker
from ai.core.monitoring.health.reporter import HealthReporter
from ai.core.monitoring.visualization.dashboard_generator import DashboardGenerator
from ai.core.monitoring.visualization.alert_visualizer import AlertVisualizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DomainMetrics:
    """
    کلاس DomainMetrics برای جمع‌آوری، تجمیع و گزارش متریک‌های مربوط به مدیریت دانش حوزه‌ای.

    ویژگی‌ها:
      - collector: شیء جمع‌آوری متریک‌ها (MetricsCollector).
      - aggregator: شیء تجمیع متریک‌ها (MetricsAggregator).
      - exporter: شیء صادرکننده متریک‌ها (MetricsExporter).
      - health_checker: شیء بررسی سلامت سیستم (HealthChecker).
      - health_reporter: شیء گزارش‌دهی سلامت سیستم (HealthReporter).
      - dashboard_generator: شیء تولید داشبورد مصورسازی (DashboardGenerator).
      - alert_visualizer: شیء نمایش هشدارهای مانیتورینگ (AlertVisualizer).

      - internal_metrics: دیکشنری متریک‌های داخلی شامل اطلاعات کلیدی:
          * requests: تعداد کل درخواست‌ها
          * cache_hits: تعداد هیت‌های کش
          * discovered_domains: تعداد حوزه‌های جدید کشف‌شده
          * discovered_concepts: تعداد مفاهیم جدید کشف‌شده
          * discovered_relations: تعداد روابط جدید کشف‌شده
          * domain_additions: تعداد حوزه‌های جدید ثبت‌شده
          * concept_additions: تعداد مفاهیم جدید ثبت‌شده
          * relation_additions: تعداد روابط جدید ثبت‌شده
          * total_processing_time: جمع زمان پردازش‌ها
          * error_count: تعداد خطاهای رخ‌داده
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
        سازنده DomainMetrics

        Args:
            collector (Optional[MetricsCollector]): شیء جمع‌آوری متریک.
            aggregator (Optional[MetricsAggregator]): شیء تجمیع متریک‌ها.
            exporter (Optional[MetricsExporter]): شیء صادرکننده متریک‌ها.
            health_checker (Optional[HealthChecker]): شیء بررسی سلامت سیستم.
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

        # دیکشنری داخلی متریک‌ها
        self.internal_metrics = {
            "requests": 0,
            "cache_hits": 0,
            "discovered_domains": 0,
            "discovered_concepts": 0,
            "discovered_relations": 0,
            "domain_additions": 0,
            "concept_additions": 0,
            "relation_additions": 0,
            "attribute_additions": 0,
            "error_count": 0,
            "total_processing_time": 0.0
        }
        self.logger.info("DomainMetrics با موفقیت مقداردهی اولیه شد.")

    def record_request(self) -> None:
        """
        ثبت یک درخواست جدید (مثلاً در شروع هر عملیات پردازشی).
        """
        self.internal_metrics["requests"] += 1
        self.logger.debug("DomainMetrics: یک درخواست جدید ثبت شد.")

    def record_cache_hit(self) -> None:
        """
        ثبت برخورد کش (cache hit).
        """
        self.internal_metrics["cache_hits"] += 1
        self.logger.debug("DomainMetrics: برخورد کش ثبت شد.")

    def record_discovered_domain(self, count: int = 1) -> None:
        """
        ثبت کشف حوزه‌های جدید.
        """
        self.internal_metrics["discovered_domains"] += count
        self.logger.debug(f"DomainMetrics: {count} حوزه جدید کشف شد.")

    def record_discovered_concept(self, count: int = 1) -> None:
        """
        ثبت کشف مفاهیم جدید.
        """
        self.internal_metrics["discovered_concepts"] += count
        self.logger.debug(f"DomainMetrics: {count} مفهوم جدید کشف شد.")

    def record_discovered_relation(self, count: int = 1) -> None:
        """
        ثبت کشف روابط جدید.
        """
        self.internal_metrics["discovered_relations"] += count
        self.logger.debug(f"DomainMetrics: {count} رابطه جدید کشف شد.")

    def record_domain_addition(self, count: int = 1) -> None:
        """
        ثبت افزودن حوزه (از طریق domain_services).
        """
        self.internal_metrics["domain_additions"] += count
        self.logger.debug(f"DomainMetrics: {count} حوزه جدید افزوده شد.")

    def record_concept_addition(self, count: int = 1) -> None:
        """
        ثبت افزودن مفهوم (از طریق domain_services).
        """
        self.internal_metrics["concept_additions"] += count
        self.logger.debug(f"DomainMetrics: {count} مفهوم جدید افزوده شد.")

    def record_relation_addition(self, count: int = 1) -> None:
        """
        ثبت افزودن رابطه جدید.
        """
        self.internal_metrics["relation_additions"] += count
        self.logger.debug(f"DomainMetrics: {count} رابطه جدید افزوده شد.")

    def record_attribute_addition(self, count: int = 1) -> None:
        """
        ثبت افزودن ویژگی جدید به مفهوم.
        """
        self.internal_metrics["attribute_additions"] += count
        self.logger.debug(f"DomainMetrics: {count} ویژگی جدید افزوده شد.")

    def record_error(self) -> None:
        """
        ثبت وقوع یک خطا.
        """
        self.internal_metrics["error_count"] += 1
        self.logger.debug("DomainMetrics: یک خطا ثبت شد.")

    def record_processing_time(self, elapsed: float) -> None:
        """
        ثبت زمان پردازش (بر حسب ثانیه).
        """
        self.internal_metrics["total_processing_time"] += elapsed
        self.logger.debug(f"DomainMetrics: زمان پردازش {elapsed:.4f} ثانیه اضافه شد.")

    def aggregate_and_export(self) -> None:
        """
        فراخوانی aggregator برای تجمیع متریک‌ها و exporter برای خروجی‌گرفتن آن‌ها.
        """
        if not self.aggregator:
            self.logger.warning("DomainMetrics: aggregator تنظیم نشده است؛ امکان تجمیع متریک‌ها وجود ندارد.")
            return

        if not self.exporter:
            self.logger.warning("DomainMetrics: exporter تنظیم نشده است؛ امکان خروجی گرفتن متریک‌ها وجود ندارد.")
            return

        try:
            self.aggregator.aggregate_metrics()
            self.logger.info("DomainMetrics: تجمیع متریک‌ها انجام شد.")
        except Exception as e:
            self.logger.error(f"DomainMetrics: خطا در تجمیع متریک‌ها: {e}")
            return

        try:
            self.exporter.export_metrics()
            self.logger.info("DomainMetrics: متریک‌ها با موفقیت خروجی گرفته شدند.")
        except Exception as e:
            self.logger.error(f"DomainMetrics: خطا در خروجی گرفتن متریک‌ها: {e}")

    def generate_dashboard(self) -> None:
        """
        فراخوانی DashboardGenerator برای تولید داشبورد مصورسازی متریک‌ها.
        """
        if not self.dashboard_generator:
            self.logger.warning("DomainMetrics: dashboard_generator تنظیم نشده است؛ امکان ساخت داشبورد وجود ندارد.")
            return

        # اینجا می‌توانید داده‌های متریک فعلی را نیز به متد generate_dashboard پاس بدهید
        try:
            self.dashboard_generator.generate_dashboard()
            self.logger.info("DomainMetrics: داشبورد تولید شد.")
        except Exception as e:
            self.logger.error(f"DomainMetrics: خطا در تولید داشبورد: {e}")

    def visualize_alerts(self) -> None:
        """
        در صورت نیاز به مصورسازی هشدارها، از AlertVisualizer استفاده می‌کند.
        """
        if not self.alert_visualizer:
            self.logger.warning("DomainMetrics: alert_visualizer تنظیم نشده است؛ امکان نمایش هشدارها وجود ندارد.")
            return

        try:
            self.alert_visualizer.visualize_alerts()
            self.logger.info("DomainMetrics: هشدارها نمایش داده شدند.")
        except Exception as e:
            self.logger.error(f"DomainMetrics: خطا در نمایش هشدارها: {e}")

    def check_system_health(self) -> Dict[str, Any]:
        """
        بررسی سلامت سیستم متریک‌ها با استفاده از HealthChecker و HealthReporter.

        Returns:
            dict: گزارشی از وضعیت سلامت سیستم
        """
        if not self.health_checker or not self.health_reporter:
            self.logger.warning("DomainMetrics: health_checker یا health_reporter تنظیم نشده‌اند.")
            return {"status": "unavailable", "details": "No health checker or reporter configured."}

        try:
            self.health_checker.run_health_checks()
            self.health_reporter.report_health()
            return {"status": "ok", "details": "Health checks performed and reported."}
        except Exception as e:
            self.logger.error(f"DomainMetrics: خطا در بررسی سلامت: {e}")
            return {"status": "error", "details": str(e)}

    def get_metrics_snapshot(self) -> Dict[str, Any]:
        """
        گرفتن نمای کلی از متریک‌های جمع‌آوری شده.

        Returns:
            dict: دیکشنری شامل اطلاعات متریک‌های داخلی
        """
        return dict(self.internal_metrics)

    def reset_metrics(self) -> None:
        """
        بازنشانی متریک‌های داخلی به مقادیر اولیه.
        """
        self.internal_metrics = {
            "requests": 0,
            "cache_hits": 0,
            "discovered_domains": 0,
            "discovered_concepts": 0,
            "discovered_relations": 0,
            "domain_additions": 0,
            "concept_additions": 0,
            "relation_additions": 0,
            "attribute_additions": 0,
            "error_count": 0,
            "total_processing_time": 0.0
        }
        self.logger.info("DomainMetrics: متریک‌ها بازنشانی شدند.")


# ----------------------------------------------------------------------------
# نمونه تست مستقل
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    dm = DomainMetrics()
    dm.record_request()
    dm.record_request()
    dm.record_cache_hit()
    dm.record_discovered_domain(2)
    dm.record_discovered_concept(3)
    dm.record_relation_addition(4)
    dm.record_processing_time(0.356)
    dm.record_error()

    print("DomainMetrics snapshot:", json.dumps(dm.get_metrics_snapshot(), ensure_ascii=False, indent=2))
    # در پروژه واقعی می‌توانید متدهای aggregator/exporter را نیز پیکربندی و تست کنید.
