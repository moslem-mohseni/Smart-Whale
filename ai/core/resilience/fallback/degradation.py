import logging
from prometheus_client import Counter
from core.resource_management.monitor.threshold_manager import ThresholdManager


class ServiceDegradation:
    def __init__(self, threshold_manager: ThresholdManager):
        """
        مدیریت کاهش سطح سرویس در صورت فشار بیش از حد روی منابع سیستم
        :param threshold_manager: نمونه‌ای از ThresholdManager برای بررسی آستانه‌های مصرف منابع
        """
        self.threshold_manager = threshold_manager
        self.logger = logging.getLogger("ServiceDegradation")

        # متریک Prometheus برای شمارش دفعات کاهش سطح سرویس
        self.degradation_events = Counter("service_degradation_events", "Total service degradation events triggered")

    def check_and_degrade(self):
        """
        بررسی مصرف منابع و اعمال کاهش سطح سرویس در صورت عبور از حد مجاز
        """
        alerts = self.threshold_manager.check_thresholds()
        degradation_actions = {}

        if alerts["cpu_alert"]:
            degradation_actions["disable_heavy_tasks"] = "مصرف CPU بالا است، پردازش‌های سنگین غیرفعال شدند."
            self.logger.warning("⚠️ کاهش سطح سرویس: پردازش‌های سنگین به‌طور موقت غیرفعال شدند.")

        if alerts["memory_alert"]:
            degradation_actions["reduce_cache_size"] = "مصرف حافظه زیاد است، اندازه کش کاهش یافت."
            self.logger.warning("⚠️ کاهش سطح سرویس: اندازه کش کاهش داده شد.")

        if alerts["gpu_alert"]:
            degradation_actions["limit_graphics_processing"] = "بار پردازشی GPU بالا است، پردازش‌های گرافیکی محدود شدند."
            self.logger.warning("⚠️ کاهش سطح سرویس: پردازش‌های گرافیکی محدود شدند.")

        if degradation_actions:
            self.degradation_events.inc()  # ثبت رخداد در Prometheus

        return degradation_actions
