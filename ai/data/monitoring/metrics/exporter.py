# exporter.py
import time
import prometheus_client
from flask import Flask, Response
from aggregator import MetricsAggregator
from collector import MetricsCollector


class MetricsExporter:
    """
    کلاس برای صادر کردن متریک‌ها به Prometheus و سایر سیستم‌های مانیتورینگ.
    """

    def __init__(self, port: int = 8000):
        """
        مقداردهی اولیه کلاس.

        :param port: پورتی که Prometheus برای دریافت متریک‌ها استفاده می‌کند.
        """
        self.port = port
        self.app = Flask(__name__)
        self.collector = MetricsCollector()
        self.aggregator = MetricsAggregator()

        # ثبت مسیر API برای Prometheus
        self.app.add_url_rule('/metrics', 'metrics', self._export_metrics)

        # تعریف متریک‌های Prometheus
        self.cpu_usage = prometheus_client.Gauge('cpu_usage', 'Average CPU Usage')
        self.memory_usage = prometheus_client.Gauge('memory_usage', 'Average Memory Usage')
        self.disk_io = prometheus_client.Gauge('disk_io', 'Average Disk I/O')
        self.network_io = prometheus_client.Gauge('network_io', 'Average Network I/O')

    def _export_metrics(self):
        """
        مسیر `/metrics` را برای Prometheus تنظیم می‌کند و مقادیر متریک‌ها را ارسال می‌کند.
        """
        aggregated_metrics = self.aggregator.get_aggregated_metrics()

        # تنظیم مقادیر متریک‌ها در Prometheus
        self.cpu_usage.set(aggregated_metrics['cpu_usage']['average'])
        self.memory_usage.set(aggregated_metrics['memory_usage']['average'])
        self.disk_io.set(aggregated_metrics['disk_io']['average'])
        self.network_io.set(aggregated_metrics['network_io']['average'])

        return Response(prometheus_client.generate_latest(), mimetype='text/plain')

    def start_exporter(self):
        """
        راه‌اندازی سرور برای ارسال متریک‌ها به Prometheus.
        """
        self.app.run(host='0.0.0.0', port=self.port)


if __name__ == "__main__":
    exporter = MetricsExporter()

    # اجرای جمع‌آوری متریک‌ها در بازه‌های زمانی مشخص
    while True:
        collected_metrics = exporter.collector.collect()
        exporter.aggregator.add_metrics(collected_metrics)
        print(f"Exporting Aggregated Metrics: {exporter.aggregator.get_aggregated_metrics()}")
        time.sleep(10)

    # راه‌اندازی سرور Prometheus
    exporter.start_exporter()
