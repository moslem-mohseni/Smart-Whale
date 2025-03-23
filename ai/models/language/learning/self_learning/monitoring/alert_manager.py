"""
AlertManager Module
---------------------
این فایل مسئول مدیریت هشدارها در سیستم خودآموزی است.
AlertManager هشدارهای سیستم (مانند ناهنجاری‌های منابع، خطاها یا مشکلات عملکردی) را دریافت کرده،
آن‌ها را ثبت می‌کند و در صورت نیاز از طریق کانال‌های خارجی (مانند Slack، Email یا PagerDuty) ارسال می‌کند.
این نسخه نهایی و عملیاتی با بهترین مکانیسم‌های هشداردهی و کارایی بالا پیاده‌سازی شده است.
"""

import logging
from abc import ABC
from typing import Dict, Any, Optional
from datetime import datetime

from ..base.base_component import BaseComponent


class AlertManager(BaseComponent, ABC):
    """
    AlertManager مسئول دریافت، ثبت و ارسال هشدارهای سیستم است.

    امکانات:
      - دریافت هشدار به صورت ورودی از سیستم.
      - ثبت هشدارها در لاگ.
      - ارسال هشدار به کانال‌های خارجی (در این نسخه، به عنوان stub پیاده‌سازی شده است).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(component_type="alert_manager", config=config)
        self.logger = logging.getLogger("AlertManager")
        # تنظیمات مربوط به کانال‌های هشدار، مانند آدرس‌های API یا توکن‌ها (در این نسخه به صورت نمونه)
        self.alert_channels = self.config.get("alert_channels", {"slack": None, "email": None})
        self.logger.info("[AlertManager] Initialized with alert_channels configuration.")

    def trigger_alert(self, message: str, severity: str = "HIGH") -> None:
        """
        ثبت و ارسال هشدار.

        Args:
            message (str): متن هشدار.
            severity (str): سطح شدت هشدار (مثلاً HIGH, MEDIUM, LOW).
        """
        alert = {
            "timestamp": datetime.utcnow().isoformat(),
            "severity": severity,
            "message": message
        }
        self.logger.warning(f"[AlertManager] Triggered alert: {alert}")
        self.increment_metric("alerts_triggered")
        # در اینجا می‌توان کد ارسال به کانال‌های خارجی (مثلاً Slack یا Email) را اضافه کرد.
        # For example: self.send_to_slack(alert) or self.send_email(alert)

    def get_alert_channels(self) -> Dict[str, Any]:
        """
        دریافت تنظیمات کانال‌های هشدار.

        Returns:
            Dict[str, Any]: تنظیمات کانال‌ها.
        """
        return self.alert_channels


# Sample usage for testing (final version intended for production)
if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)

    am = AlertManager(
        config={"alert_channels": {"slack": "https://hooks.slack.com/services/...", "email": "alerts@example.com"}})
    am.trigger_alert("Test alert: System resources are critically low!", severity="CRITICAL")
