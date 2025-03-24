from abc import ABC, abstractmethod
from typing import Dict, Any


class AlertClassifier(ABC):
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


class AlertClassifier(ABC):
    """
    دسته‌بندی هشدارهای سیستم در ماژول Balance.
    """

    @abstractmethod
    def classify_alerts(self, alert_data: Dict[str, Any]) -> str:
        """
        دسته‌بندی هشدارها بر اساس سطح اهمیت.

        :param alert_data: داده‌های هشدار
        :return: سطح هشدار (کم، متوسط، بحرانی)
        """
        pass

    @abstractmethod
    def log_alerts(self, alert_data: Dict[str, Any]) -> None:
        """
        ثبت هشدارهای دسته‌بندی شده برای مانیتورینگ و تحلیل.

        :param alert_data: داده‌های هشدار
        """
        pass
