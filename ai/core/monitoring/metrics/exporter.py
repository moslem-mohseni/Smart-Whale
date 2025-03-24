import logging
import requests
from prometheus_client import push_to_gateway
from ai.core.monitoring.metrics.collector import MetricsCollector


class MetricsExporter:
    def __init__(self, metrics_collector: MetricsCollector, prometheus_gateway=None, logstash_url=None):
        """
        ارسال متریک‌های سیستم به سرویس‌های خارجی مانند Prometheus و Logstash
        :param metrics_collector: نمونه‌ای از MetricsCollector برای دریافت متریک‌های سیستم
        :param prometheus_gateway: آدرس Push Gateway برای Prometheus
        :param logstash_url: آدرس Logstash برای ارسال متریک‌ها
        """
        self.metrics_collector = metrics_collector
        self.prometheus_gateway = prometheus_gateway
        self.logstash_url = logstash_url
        self.logger = logging.getLogger("MetricsExporter")

    def export_to_prometheus(self):
        """ ارسال متریک‌ها به Push Gateway در Prometheus """
        if self.prometheus_gateway:
            try:
                push_to_gateway(self.prometheus_gateway, job="system_metrics", registry=self.metrics_collector)
                self.logger.info(f"✅ متریک‌ها به Prometheus Push Gateway ارسال شدند ({self.prometheus_gateway}).")
            except Exception as e:
                self.logger.error(f"❌ خطا در ارسال متریک‌ها به Prometheus: {e}")

    def export_to_logstash(self):
        """ ارسال متریک‌ها به Logstash برای تجزیه و تحلیل """
        if self.logstash_url:
            data = {
                "cpu_usage": self.metrics_collector.cpu_usage.collect()[0].samples[0].value,
                "memory_usage": self.metrics_collector.memory_usage.collect()[0].samples[0].value,
                "total_requests": self.metrics_collector.request_count.collect()[0].samples[0].value,
                "total_errors": self.metrics_collector.error_count.collect()[0].samples[0].value
            }

            try:
                response = requests.post(self.logstash_url, json=data)
                if response.status_code == 200:
                    self.logger.info("✅ متریک‌ها به Logstash ارسال شدند.")
                else:
                    self.logger.error(f"❌ خطا در ارسال متریک‌ها به Logstash: {response.status_code}")
            except Exception as e:
                self.logger.error(f"❌ خطا در ارسال متریک‌ها به Logstash: {e}")

    def export_metrics(self):
        """ اجرای ارسال متریک‌ها به تمام سرویس‌های موجود """
        self.export_to_prometheus()
        self.export_to_logstash()
