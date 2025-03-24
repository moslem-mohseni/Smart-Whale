from abc import ABC, abstractmethod
from typing import Dict, Any


class PerformanceMetrics(ABC):
    """
    مدیریت متریک‌های کارایی سیستم در ماژول Balance.
    """

    @abstractmethod
    def collect_performance_data(self) -> Dict[str, Any]:
        """
        جمع‌آوری داده‌های متریک‌های کارایی.

        :return: داده‌های متریک‌های عملکردی
        """
        pass

    @abstractmethod
    def analyze_performance_trends(self, metrics_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        تحلیل روندهای متریک‌های کارایی.

        :param metrics_data: داده‌های متریک عملکردی
        :return: تحلیل‌های انجام‌شده روی روندهای متریک
        """
        pass


class QualityMetrics(ABC):
    """
    مدیریت متریک‌های کیفیت داده در ماژول Balance.
    """

    @abstractmethod
    def evaluate_data_quality(self) -> Dict[str, Any]:
        """
        ارزیابی کیفیت داده‌های پردازش‌شده.

        :return: داده‌های ارزیابی‌شده کیفیت
        """
        pass

    @abstractmethod
    def monitor_quality_trends(self, quality_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        پایش روندهای کیفیت داده.

        :param quality_data: داده‌های کیفیت پردازش‌شده
        :return: تحلیل‌های انجام‌شده روی روندهای کیفیت
        """
        pass
