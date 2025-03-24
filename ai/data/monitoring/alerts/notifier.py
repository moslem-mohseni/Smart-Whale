# notifier.py
import logging
from typing import List, Dict

class AlertNotifier:
    """
    کلاس برای ارسال هشدارها به کانال‌های مختلف مانند لاگ، ایمیل، پیامک یا Kafka.
    """

    def __init__(self, channels: Dict[str, bool] = None):
        """
        مقداردهی اولیه کلاس.

        :param channels: دیکشنری شامل تنظیمات ارسال هشدارها به کانال‌های مختلف.
        """
        self.channels = channels if channels else {
            "log": True,       # ثبت هشدار در لاگ سیستم
            "email": False,    # ارسال هشدار از طریق ایمیل
            "sms": False,      # ارسال هشدار از طریق پیامک
            "kafka": False     # ارسال هشدار به Kafka
        }
        logging.basicConfig(level=logging.INFO)

    def send_alerts(self, alerts: List[str]) -> None:
        """
        ارسال هشدارها به کانال‌های فعال.

        :param alerts: لیستی از هشدارهای شناسایی‌شده.
        """
        if not alerts:
            logging.info("✅ هیچ هشداری برای ارسال وجود ندارد.")
            return

        for alert in alerts:
            if self.channels.get("log"):
                self._log_alert(alert)
            if self.channels.get("email"):
                self._send_email(alert)
            if self.channels.get("sms"):
                self._send_sms(alert)
            if self.channels.get("kafka"):
                self._send_to_kafka(alert)

    def _log_alert(self, alert: str) -> None:
        """ثبت هشدار در لاگ سیستم."""
        logging.warning(f"[LOG ALERT] {alert}")

    def _send_email(self, alert: str) -> None:
        """ارسال هشدار از طریق ایمیل (به‌صورت نمونه)."""
        logging.info(f"[EMAIL ALERT] ارسال هشدار از طریق ایمیل: {alert}")

    def _send_sms(self, alert: str) -> None:
        """ارسال هشدار از طریق پیامک (به‌صورت نمونه)."""
        logging.info(f"[SMS ALERT] ارسال هشدار از طریق پیامک: {alert}")

    def _send_to_kafka(self, alert: str) -> None:
        """ارسال هشدار به Kafka (به‌صورت نمونه)."""
        logging.info(f"[KAFKA ALERT] ارسال هشدار به Kafka: {alert}")

if __name__ == "__main__":
    notifier = AlertNotifier(channels={"log": True, "email": True, "sms": False, "kafka": False})

    # هشدارهای نمونه برای تست
    sample_alerts = [
        "🚨 هشدار: استفاده از CPU بیش از حد مجاز است! مقدار: 90%",
        "🚨 هشدار: حافظه سیستم در حال پر شدن است! مقدار: 95%"
    ]

    notifier.send_alerts(sample_alerts)
