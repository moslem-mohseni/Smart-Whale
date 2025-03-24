import time
import logging
from prometheus_client import Summary
from ai.core.monitoring.metrics.collector import MetricsCollector

class MetricsAggregator:
    def __init__(self, metrics_collector: MetricsCollector, aggregation_interval=10):
        """
        تجمیع داده‌های مانیتورینگ و ارائه تحلیل‌های پیشرفته
        :param metrics_collector: نمونه‌ای از MetricsCollector برای دریافت داده‌های خام
        :param aggregation_interval: فاصله زمانی بین پردازش داده‌ها (بر حسب ثانیه)
        """
        self.metrics_collector = metrics_collector
        self.aggregation_interval = aggregation_interval
        self.logger = logging.getLogger("MetricsAggregator")

        # متریک‌های Prometheus برای داده‌های تجمیعی
        self.avg_cpu_usage = Summary("aggregated_cpu_usage", "Aggregated CPU usage over time")
        self.avg_memory_usage = Summary("aggregated_memory_usage", "Aggregated Memory usage over time")
        self.error_rate = Summary("error_rate", "Error rate based on total requests")

    def aggregate_metrics(self):
        """ تجمیع داده‌های مانیتورینگ برای تحلیل بهتر """
        cpu_usage = self.metrics_collector.cpu_usage.collect()[0].samples[0].value
        memory_usage = self.metrics_collector.memory_usage.collect()[0].samples[0].value
        total_requests = self.metrics_collector.request_count.collect()[0].samples[0].value
        total_errors = self.metrics_collector.error_count.collect()[0].samples[0].value

        error_rate = (total_errors / total_requests) * 100 if total_requests > 0 else 0

        # ثبت داده‌های تجمیعی در Prometheus
        self.avg_cpu_usage.observe(cpu_usage)
        self.avg_memory_usage.observe(memory_usage)
        self.error_rate.observe(error_rate)

        self.logger.info(f"🔹 میانگین مصرف CPU: {cpu_usage:.2f}% | میانگین مصرف حافظه: {memory_usage:.2f}% | نرخ خطا: {error_rate:.2f}%")

    def start_aggregation(self):
        """ اجرای فرآیند تجمیع داده‌ها در بازه‌های زمانی مشخص """
        while True:
            self.aggregate_metrics()
            time.sleep(self.aggregation_interval)
