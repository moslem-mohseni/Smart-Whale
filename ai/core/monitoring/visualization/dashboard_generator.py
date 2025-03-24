import logging
import requests
import matplotlib.pyplot as plt
from datetime import datetime
from prometheus_client import Gauge
from ai.core.monitoring.metrics.collector import MetricsCollector


class DashboardGenerator:
    def __init__(self, metrics_collector: MetricsCollector, prometheus_url=None):
        """
        ایجاد داشبورد گرافیکی برای نمایش متریک‌های سیستم
        :param metrics_collector: نمونه‌ای از MetricsCollector برای دریافت داده‌های متریک
        :param prometheus_url: آدرس Prometheus برای دریافت متریک‌ها
        """
        self.metrics_collector = metrics_collector
        self.prometheus_url = prometheus_url
        self.logger = logging.getLogger("DashboardGenerator")

    def fetch_data_from_prometheus(self, metric_name):
        """ دریافت داده‌های متریک از Prometheus """
        if not self.prometheus_url:
            self.logger.warning("❌ Prometheus URL تنظیم نشده است.")
            return None

        query_url = f"{self.prometheus_url}/api/v1/query?query={metric_name}"
        try:
            response = requests.get(query_url)
            if response.status_code == 200:
                return response.json().get("data", {}).get("result", [])
            else:
                self.logger.error(f"⚠️ خطا در دریافت متریک {metric_name} از Prometheus: {response.status_code}")
        except Exception as e:
            self.logger.error(f"❌ خطا در برقراری ارتباط با Prometheus: {e}")

        return None

    def generate_local_dashboard(self):
        """ تولید نمودارهای متریک به‌صورت محلی با Matplotlib """
        cpu_usage = self.metrics_collector.cpu_usage.collect()[0].samples[0].value
        memory_usage = self.metrics_collector.memory_usage.collect()[0].samples[0].value
        total_requests = self.metrics_collector.request_count.collect()[0].samples[0].value
        total_errors = self.metrics_collector.error_count.collect()[0].samples[0].value

        labels = ["CPU Usage (%)", "Memory Usage (%)", "Total Requests", "Total Errors"]
        values = [cpu_usage, memory_usage, total_requests, total_errors]

        plt.figure(figsize=(8, 5))
        plt.bar(labels, values, color=["blue", "green", "orange", "red"])
        plt.xlabel("Metrics")
        plt.ylabel("Value")
        plt.title(f"System Metrics - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        plt.savefig("dashboard_metrics.png")
        plt.close()

        self.logger.info("📊 داشبورد متریک‌های سیستم به‌صورت PNG ذخیره شد.")

    def generate_dashboard(self):
        """ اجرای ایجاد داشبورد بر اساس وضعیت Prometheus """
        if self.prometheus_url:
            self.logger.info("📡 دریافت داده‌ها از Prometheus و ارسال به Grafana...")
            for metric in ["system_cpu_usage", "system_memory_usage", "system_request_count", "system_error_count"]:
                data = self.fetch_data_from_prometheus(metric)
                if data:
                    self.logger.info(f"✅ داده‌های متریک {metric} از Prometheus دریافت شد.")
                else:
                    self.logger.warning(f"⚠️ داده‌ای برای {metric} در Prometheus یافت نشد.")

        self.logger.info("🖥️ تولید داشبورد گرافیکی متریک‌ها به‌صورت محلی...")
        self.generate_local_dashboard()
