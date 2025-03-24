# detector.py
import logging
from typing import Dict

class AlertDetector:
    """
    کلاس برای تشخیص ناهنجاری‌ها و تولید هشدارها بر اساس متریک‌های سیستم.
    """

    def __init__(self, thresholds: Dict[str, float] = None):
        """
        مقداردهی اولیه کلاس.

        :param thresholds: دیکشنری شامل آستانه‌های هشدار برای متریک‌های مختلف.
        """
        self.thresholds = thresholds if thresholds else {
            "cpu_usage": 85.0,  # هشدار اگر استفاده از CPU بیشتر از 85% شود
            "memory_usage": 90.0,  # هشدار اگر استفاده از حافظه بیشتر از 90% شود
            "disk_io": 150.0,  # هشدار برای عملیات بالای دیسک
            "network_io": 200.0  # هشدار برای ترافیک شبکه غیرعادی
        }
        self.alerts = []
        logging.basicConfig(level=logging.INFO)

    def check_for_alerts(self, metrics_data: Dict[str, float]) -> list:
        """
        بررسی متریک‌ها و ایجاد هشدارها در صورت وجود ناهنجاری.

        :param metrics_data: دیکشنری شامل متریک‌های سیستم.
        :return: لیستی از هشدارهای شناسایی‌شده.
        """
        self.alerts.clear()
        for key, value in metrics_data.items():
            if key in self.thresholds and value >= self.thresholds[key]:
                alert_msg = f"🚨 هشدار: مقدار {key} بیش از حد مجاز است! مقدار فعلی: {value}% (حد آستانه: {self.thresholds[key]}%)"
                self.alerts.append(alert_msg)
                logging.warning(alert_msg)

        return self.alerts

if __name__ == "__main__":
    detector = AlertDetector()

    # شبیه‌سازی متریک‌های نمونه
    test_metrics = {
        "cpu_usage": 87.5,
        "memory_usage": 92.3,
        "disk_io": 120.0,
        "network_io": 210.0
    }

    alerts = detector.check_for_alerts(test_metrics)
    if alerts:
        print("\n".join(alerts))
    else:
        print("✅ هیچ ناهنجاری‌ای شناسایی نشد.")
