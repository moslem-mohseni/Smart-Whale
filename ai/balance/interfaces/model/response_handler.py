from abc import ABC, abstractmethod
from typing import Any, Dict

class ResponseHandler(ABC):
    """
    مدیریت پاسخ‌های پردازش‌شده برای مدل‌ها.
    """

    @abstractmethod
    def format_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        فرمت‌بندی پاسخ قبل از ارسال به مدل‌ها.

        :param response_data: داده‌های پاسخ خام
        :return: داده‌های پاسخ فرمت شده
        """
        pass

    @abstractmethod
    def log_response(self, model_id: str, response_data: Dict[str, Any]) -> None:
        """
        ثبت پاسخ‌های دریافتی برای مانیتورینگ و تحلیل.

        :param model_id: شناسه مدل
        :param response_data: داده‌های پاسخ دریافتی
        """
        pass