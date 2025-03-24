import logging
from core.resource_management.monitor.threshold_manager import ThresholdManager
from prometheus_client import Counter

class AlertGenerator:
    def __init__(self, threshold_manager: ThresholdManager):
        """
        مدیریت و ارسال هشدارها برای مصرف بیش از حد منابع
        :param threshold_manager: نمونه‌ای از ThresholdManager برای بررسی وضعیت منابع
        """
        self.threshold_manager = threshold_manager
        self.logger = logging.getLogger("ResourceAlerts")

        # تعریف شمارنده هشدارها در Prometheus
        self.cpu_alerts = Counter("cpu_alerts_total", "Total CPU threshold alerts triggered")
        self.memory_alerts = Counter("memory_alerts_total", "Total memory threshold alerts triggered")
        self.gpu_alerts = Counter("gpu_alerts_total", "Total GPU threshold alerts triggered")

    def check_and_alert(self):
        """
        بررسی مصرف منابع و ارسال هشدار در صورت نیاز
        """
        alerts = self.threshold_manager.check_thresholds()

        if alerts["cpu_alert"]:
            self.cpu_alerts.inc()  # افزایش شمارنده هشدارهای CPU در Prometheus
            self.logger.warning("⚠️ مصرف CPU از حد مجاز فراتر رفت!")

        if alerts["memory_alert"]:
            self.memory_alerts.inc()  # افزایش شمارنده هشدارهای RAM در Prometheus
            self.logger.warning("⚠️ مصرف حافظه از حد مجاز فراتر رفت!")

        if alerts["gpu_alert"]:
            self.gpu_alerts.inc()  # افزایش شمارنده هشدارهای GPU در Prometheus
            self.logger.warning("⚠️ مصرف GPU از حد مجاز فراتر رفت!")
