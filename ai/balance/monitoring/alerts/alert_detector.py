from abc import ABC, abstractmethod
from typing import Dict, Any


class AlertDetector(ABC):
    """
    تشخیص هشدارها در ماژول Balance.
    """

    @abstractmethod
    def detect_anomalies(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        شناسایی ناهنجاری‌ها در داده‌های سیستم.

        :param system_data: داده‌های مانیتور شده سیستم
        :return: هشدارهای تشخیص داده شده
        """
        pass

    @abstractmethod
    def trigger_alerts(self, alert_data: Dict[str, Any]) -> None:
        """
        ارسال هشدارهای سیستم بر اساس ناهنجاری‌های تشخیص داده شده.

        :param alert_data: داده‌های هشدار
        """
        pass
