from abc import ABC, abstractmethod
from typing import Dict, Any

class TrendAnalyzer(ABC):
    """
    تحلیل روند داده‌ها و عملکرد سیستم در ماژول Balance.
    """

    @abstractmethod
    def analyze_trends(self, historical_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        تحلیل روندهای گذشته برای شناسایی الگوهای تغییر.

        :param historical_data: داده‌های تاریخی مورد بررسی
        :return: تحلیل‌های انجام‌شده روی روندهای سیستم
        """
        pass

    @abstractmethod
    def predict_future_trends(self, trend_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        پیش‌بینی روندهای آینده بر اساس داده‌های تحلیلی.

        :param trend_data: داده‌های پردازش‌شده مرتبط با روندها
        :return: پیش‌بینی روندهای آینده
        """
        pass
