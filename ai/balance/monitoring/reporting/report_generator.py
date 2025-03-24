from abc import ABC, abstractmethod
from typing import Dict, Any

class ReportGenerator(ABC):
    """
    تولید گزارش‌های آماری و عملیاتی در ماژول Balance.
    """

    @abstractmethod
    def generate_report(self, report_data: Dict[str, Any]) -> str:
        """
        تولید گزارش از داده‌های پردازشی.

        :param report_data: داده‌های مورد نیاز برای گزارش
        :return: گزارش تولید شده به عنوان یک رشته
        """
        pass

    @abstractmethod
    def export_report(self, report: str, format_type: str) -> None:
        """
        صادر کردن گزارش در قالب موردنظر (مثلاً JSON، CSV، PDF).

        :param report: گزارش تولید شده
        :param format_type: نوع فرمت خروجی (JSON, CSV, PDF)
        """
        pass
