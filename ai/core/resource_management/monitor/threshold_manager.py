import psutil
import torch
from prometheus_client import Gauge

class ThresholdManager:
    def __init__(self, cpu_threshold=80, memory_threshold=75, gpu_threshold=90):
        """
        مدیریت آستانه‌های مصرف منابع و ارسال هشدار به Prometheus
        :param cpu_threshold: حد آستانه مصرف CPU (پیش‌فرض 80٪)
        :param memory_threshold: حد آستانه مصرف حافظه (پیش‌فرض 75٪)
        :param gpu_threshold: حد آستانه مصرف GPU (پیش‌فرض 90٪)
        """
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.gpu_threshold = gpu_threshold

        # تعریف متریک‌های Prometheus برای هشدارها
        self.cpu_alert = Gauge("cpu_threshold_alert", "CPU usage exceeded threshold")
        self.memory_alert = Gauge("memory_threshold_alert", "Memory usage exceeded threshold")
        self.gpu_alert = Gauge("gpu_threshold_alert", "GPU usage exceeded threshold")

    def check_thresholds(self):
        """ بررسی مصرف منابع و فعال‌سازی هشدارها در صورت عبور از حد مجاز """
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent
        gpu_usage = 0  # مقدار پیش‌فرض اگر GPU وجود نداشته باشد

        if torch.cuda.is_available():
            gpu_usage = (torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_memory) * 100

        # بررسی عبور از آستانه‌ها
        self.cpu_alert.set(1 if cpu_usage > self.cpu_threshold else 0)
        self.memory_alert.set(1 if memory_usage > self.memory_threshold else 0)
        self.gpu_alert.set(1 if gpu_usage > self.gpu_threshold else 0)

        return {
            "cpu_alert": cpu_usage > self.cpu_threshold,
            "memory_alert": memory_usage > self.memory_threshold,
            "gpu_alert": gpu_usage > self.gpu_threshold
        }
