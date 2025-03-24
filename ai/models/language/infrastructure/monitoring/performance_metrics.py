import logging
from ai.core.monitoring.metrics.collector import MetricsCollector
from ai.core.monitoring.metrics.aggregator import MetricsAggregator
from ai.core.monitoring.metrics.exporter import MetricsExporter
from ai.models.language.infrastructure.monitoring.health_check import HealthCheck

class PerformanceMetrics:
    """
    این کلاس وظیفه‌ی جمع‌آوری، پردازش، و ارسال متریک‌های عملکردی پردازش زبان را بر عهده دارد.
    """

    def __init__(self, metrics_collector: MetricsCollector, metrics_aggregator: MetricsAggregator, metrics_exporter: MetricsExporter, health_check: HealthCheck):
        self.metrics_collector = metrics_collector
        self.metrics_aggregator = metrics_aggregator
        self.metrics_exporter = metrics_exporter
        self.health_check = health_check
        logging.info("✅ PerformanceMetrics مقداردهی شد و ارتباط با سیستم‌های مانیتورینگ برقرار شد.")

    async def collect_metrics(self):
        """
        جمع‌آوری متریک‌های عملکردی پردازش زبان.
        """
        try:
            metrics_data = await self.metrics_collector.collect()
            aggregated_data = await self.metrics_aggregator.aggregate(metrics_data)
            await self.metrics_exporter.export(aggregated_data)
            logging.info("📊 متریک‌های عملکردی جمع‌آوری، پردازش و ارسال شدند.")
        except Exception as e:
            logging.error(f"❌ خطا در جمع‌آوری متریک‌های عملکردی: {e}")

    async def get_system_health_metrics(self):
        """
        دریافت متریک‌های سلامت سیستم از `HealthCheck` و پردازش آن‌ها.

        :return: دیکشنری شامل وضعیت سلامت سیستم و متریک‌های مربوطه.
        """
        try:
            health_data = await self.health_check.check_all_services()
            logging.info("📊 متریک‌های سلامت سیستم پردازش شدند.")
            return health_data
        except Exception as e:
            logging.error(f"❌ خطا در دریافت متریک‌های سلامت سیستم: {e}")
            return {"status": "error", "message": str(e)}
