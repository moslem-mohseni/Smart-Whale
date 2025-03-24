from abc import ABC, abstractmethod
from typing import Any, Dict

class ModelInterface(ABC):
    """
    رابط اصلی برای تعامل ماژول Balance با مدل‌ها.
    """

    @abstractmethod
    def send_request(self, model_id: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        ارسال درخواست به مدل مشخص شده.

        :param model_id: شناسه مدل
        :param request_data: داده‌های درخواست
        :return: پاسخ پردازش شده از مدل
        """
        pass

    @abstractmethod
    def receive_response(self, model_id: str, response_data: Dict[str, Any]) -> None:
        """
        دریافت و پردازش پاسخ از مدل.

        :param model_id: شناسه مدل
        :param response_data: داده‌های دریافتی از مدل
        """
        pass
