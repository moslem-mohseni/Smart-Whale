from abc import ABC, abstractmethod
from typing import Any, Dict

class RequestHandler(ABC):
    """
    مدیریت درخواست‌های دریافتی از مدل‌ها.
    """

    @abstractmethod
    def validate_request(self, request_data: Dict[str, Any]) -> bool:
        """
        اعتبارسنجی درخواست‌های ورودی از مدل‌ها.

        :param request_data: داده‌های درخواست
        :return: نتیجه اعتبارسنجی (True/False)
        """
        pass

    @abstractmethod
    def preprocess_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        پردازش اولیه روی داده‌های درخواست برای بهینه‌سازی پردازش.

        :param request_data: داده‌های درخواست
        :return: داده‌های پردازش‌شده
        """
        pass
