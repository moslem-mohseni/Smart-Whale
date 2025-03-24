import time
import psutil
import logging
from prometheus_client import Gauge, Counter, start_http_server


class MetricsCollector:
    def __init__(self, port=9100, update_interval=5):
        """
        جمع‌آوری متریک‌های سیستم و ارسال به Prometheus
        :param port: پورتی که سرور متریک‌ها روی آن اجرا می‌شود
        :param update_interval: فاصله به‌روزرسانی متریک‌ها (بر حسب ثانیه)
        """
        self.logger = logging.getLogger("MetricsCollector")
        self.update_interval = update_interval

        # تعریف متریک‌های Prometheus
        self.cpu_usage = Gauge("system_cpu_usage", "Current CPU usage percentage")
        self.memory_usage = Gauge("system_memory_usage", "Current memory usage percentage")
        self.request_count = Counter("system_request_count", "Total number of processed requests")
        self.error_count = Counter("system_error_count", "Total number of errors encountered")

        # شروع سرور متریک‌ها
        start_http_server(port)
        self.logger.info(f"✅ سرور Prometheus برای متریک‌ها روی پورت {port} اجرا شد.")

    def collect_metrics(self):
        """ جمع‌آوری و به‌روزرسانی متریک‌های سیستم """
        self.cpu_usage.set(psutil.cpu_percent(interval=1))
        self.memory_usage.set(psutil.virtual_memory().percent)

    def increment_request_count(self):
        """ افزایش شمارنده درخواست‌ها """
        self.request_count.inc()

    def increment_error_count(self):
        """ افزایش شمارنده خطاها """
        self.error_count.inc()

    def start_monitoring(self):
        """ اجرای فرآیند مانیتورینگ در بازه‌های زمانی مشخص """
        while True:
            self.collect_metrics()
            time.sleep(self.update_interval)
