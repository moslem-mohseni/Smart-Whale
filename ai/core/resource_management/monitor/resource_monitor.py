import psutil
import torch
from prometheus_client import Gauge, start_http_server

class ResourceMonitor:
    def __init__(self, port=8000):
        """
        مانیتورینگ مصرف منابع سیستم و ارسال متریک‌ها به Prometheus
        :param port: پورت برای سرور متریک‌های Prometheus
        """
        self.cpu_usage = Gauge("system_cpu_usage", "CPU usage percentage")
        self.memory_usage = Gauge("system_memory_usage", "Memory usage percentage")
        self.gpu_usage = Gauge("system_gpu_usage", "GPU usage percentage")

        # شروع سرور متریک‌ها
        start_http_server(port)

    def collect_metrics(self):
        """ جمع‌آوری متریک‌های CPU، RAM و GPU """
        self.cpu_usage.set(psutil.cpu_percent(interval=1))
        self.memory_usage.set(psutil.virtual_memory().percent)

        if torch.cuda.is_available():
            gpu_usage_percent = (torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_memory) * 100
            self.gpu_usage.set(gpu_usage_percent)
        else:
            self.gpu_usage.set(0)

    def start_monitoring(self, interval=5):
        """ اجرای مانیتورینگ به‌صورت دوره‌ای """
        import time
        while True:
            self.collect_metrics()
            time.sleep(interval)
