import logging
import psutil
import requests
import torch
from prometheus_client import Gauge


class HealthChecker:
    def __init__(self, services=None):
        """
        بررسی سلامت سیستم و سرویس‌های کلیدی
        :param services: لیستی از URLهای API برای بررسی وضعیت سرویس‌های خارجی
        """
        self.logger = logging.getLogger("HealthChecker")
        self.services = services or {}

        # متریک‌های سلامت در Prometheus
        self.cpu_health = Gauge("system_cpu_health", "CPU health status (1 = Healthy, 0 = Unhealthy)")
        self.memory_health = Gauge("system_memory_health", "Memory health status (1 = Healthy, 0 = Unhealthy)")
        self.gpu_health = Gauge("system_gpu_health", "GPU health status (1 = Healthy, 0 = Unhealthy)")
        self.service_health = Gauge("external_service_health", "Health status of external services", ["service"])

    def check_system_health(self):
        """ بررسی سلامت CPU، حافظه و GPU """
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent

        # بررسی سلامت CPU
        if cpu_usage < 85:  # اگر کمتر از 85% باشد، سالم در نظر گرفته می‌شود
            self.cpu_health.set(1)
        else:
            self.cpu_health.set(0)
            self.logger.warning(f"⚠️ مصرف CPU بالاست ({cpu_usage}%).")

        # بررسی سلامت حافظه
        if memory_usage < 80:
            self.memory_health.set(1)
        else:
            self.memory_health.set(0)
            self.logger.warning(f"⚠️ مصرف حافظه بالاست ({memory_usage}%).")

        # بررسی سلامت GPU در صورت موجود بودن
        if torch.cuda.is_available():
            gpu_usage = (torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_memory) * 100
            if gpu_usage < 90:
                self.gpu_health.set(1)
            else:
                self.gpu_health.set(0)
                self.logger.warning(f"⚠️ مصرف GPU بالاست ({gpu_usage:.2f}%).")
        else:
            self.gpu_health.set(0)

    def check_service_health(self):
        """ بررسی سلامت APIها و سرویس‌های خارجی """
        for service, url in self.services.items():
            try:
                response = requests.get(url, timeout=3)
                if response.status_code == 200:
                    self.service_health.labels(service=service).set(1)
                    self.logger.info(f"✅ سرویس {service} سالم است.")
                else:
                    self.service_health.labels(service=service).set(0)
                    self.logger.warning(f"⚠️ مشکل در سرویس {service} (کد {response.status_code}).")
            except requests.RequestException as e:
                self.service_health.labels(service=service).set(0)
                self.logger.error(f"❌ عدم دسترسی به سرویس {service}: {e}")

    def run_health_checks(self):
        """ اجرای بررسی‌های سلامت سیستم و سرویس‌ها """
        self.check_system_health()
        self.check_service_health()
