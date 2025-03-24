from abc import ABC, abstractmethod
from typing import Any, Dict

class APIInterface(ABC):
    """
    رابط API برای تعامل با سرویس‌های خارجی در ماژول Balance.
    """

    @abstractmethod
    def send_request(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        ارسال درخواست به یک API خارجی.

        :param endpoint: آدرس API
        :param payload: داده‌های ارسال‌شده
        :return: پاسخ دریافت‌شده از API
        """
        pass

    @abstractmethod
    def receive_response(self, response_data: Dict[str, Any]) -> None:
        """
        پردازش پاسخ دریافتی از API خارجی.

        :param response_data: داده‌های دریافت‌شده
        """
        pass
