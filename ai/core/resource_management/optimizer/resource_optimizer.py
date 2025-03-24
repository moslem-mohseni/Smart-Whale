import logging
from ai.core.resource_management.monitor.resource_monitor import ResourceMonitor
from ai.core.resource_management.monitor.threshold_manager import ThresholdManager


class ResourceOptimizer:
    def __init__(self, resource_monitor: ResourceMonitor, threshold_manager: ThresholdManager):
        """
        بهینه‌سازی مصرف منابع بر اساس داده‌های مانیتورینگ و آستانه‌های تعیین‌شده
        :param resource_monitor: نمونه‌ای از ResourceMonitor برای بررسی مصرف منابع
        :param threshold_manager: نمونه‌ای از ThresholdManager برای بررسی عبور از حد مجاز
        """
        self.resource_monitor = resource_monitor
        self.threshold_manager = threshold_manager
        self.logger = logging.getLogger("ResourceOptimizer")

    def analyze_and_optimize(self):
        """
        بررسی میزان مصرف منابع و ارائه پیشنهادات بهینه‌سازی
        :return: دیکشنری شامل پیشنهادات بهینه‌سازی
        """
        # دریافت وضعیت منابع
        cpu_usage = self.resource_monitor.cpu_usage.collect()[0].samples[0].value
        memory_usage = self.resource_monitor.memory_usage.collect()[0].samples[0].value
        gpu_usage = self.resource_monitor.gpu_usage.collect()[0].samples[0].value if self.resource_monitor.gpu_usage else 0

        # دریافت وضعیت هشدارها
        alerts = self.threshold_manager.check_thresholds()
        suggestions = {}

        if alerts["cpu_alert"]:
            suggestions["reduce_cpu_tasks"] = "مصرف CPU زیاد است. پیشنهاد می‌شود تعداد پردازش‌های سنگین کاهش یابد."
            self.logger.warning("⚠️ پیشنهاد بهینه‌سازی: کاهش پردازش‌های سنگین CPU.")

        if alerts["memory_alert"]:
            suggestions["reduce_memory_usage"] = "مصرف حافظه بیش از حد است. پیشنهاد می‌شود اندازه کش کاهش یابد."
            self.logger.warning("⚠️ پیشنهاد بهینه‌سازی: کاهش میزان مصرف حافظه کش.")

        if alerts["gpu_alert"]:
            suggestions["optimize_gpu_tasks"] = "بار پردازشی GPU زیاد است. پیشنهاد می‌شود مدل‌های پیچیده با GPU پایین‌تر اجرا شوند."
            self.logger.warning("⚠️ پیشنهاد بهینه‌سازی: کاهش حجم پردازش‌های گرافیکی سنگین.")

        return suggestions
