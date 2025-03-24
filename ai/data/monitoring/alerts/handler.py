# handler.py
import logging
import time
from typing import List, Dict
from detector import AlertDetector
from notifier import AlertNotifier


class AlertHandler:
    """
    کلاس برای مدیریت هشدارها، جلوگیری از هشدارهای تکراری و اجرای سیاست‌های رسیدگی.
    """

    def __init__(self, cooldown_period: int = 60):
        """
        مقداردهی اولیه کلاس.

        :param cooldown_period: مدت زمان خنک‌سازی (Cooldown) برای جلوگیری از هشدارهای تکراری.
        """
        self.cooldown_period = cooldown_period
        self.last_alert_times: Dict[str, float] = {}
        self.detector = AlertDetector()
        self.notifier = AlertNotifier()
        logging.basicConfig(level=logging.INFO)

    def process_alerts(self, metrics_data: Dict[str, float]) -> None:
        """
        بررسی متریک‌ها، شناسایی هشدارها و ارسال هشدارهای جدید.

        :param metrics_data: دیکشنری شامل متریک‌های جدید سیستم.
        """
        alerts = self.detector.check_for_alerts(metrics_data)
        new_alerts = self._filter_duplicate_alerts(alerts)

        if new_alerts:
            self.notifier.send_alerts(new_alerts)
        else:
            logging.info("✅ هیچ هشداری ارسال نشد. تمام هشدارها در بازه خنک‌سازی هستند.")

    def _filter_duplicate_alerts(self, alerts: List[str]) -> List[str]:
        """
        حذف هشدارهای تکراری که در دوره خنک‌سازی قرار دارند.

        :param alerts: لیستی از هشدارهای جدید.
        :return: لیستی از هشدارهایی که نیاز به ارسال دارند.
        """
        current_time = time.time()
        new_alerts = []

        for alert in alerts:
            last_sent_time = self.last_alert_times.get(alert, 0)
            if current_time - last_sent_time >= self.cooldown_period:
                new_alerts.append(alert)
                self.last_alert_times[alert] = current_time
            else:
                logging.info(f"⚠️ هشدار تکراری شناسایی شد و ارسال نمی‌شود: {alert}")

        return new_alerts


if __name__ == "__main__":
    handler = AlertHandler(cooldown_period=60)

    # متریک‌های نمونه برای تست
    test_metrics = {
        "cpu_usage": 90.0,
        "memory_usage": 92.3,
        "disk_io": 140.0,
        "network_io": 220.0
    }

    # پردازش هشدارها
    handler.process_alerts(test_metrics)
