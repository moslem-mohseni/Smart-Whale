from abc import ABC, abstractmethod
from typing import Any, Dict

class DataInterface(ABC):
    """
    رابط اصلی برای تعامل ماژول Balance با داده‌ها.
    """

    @abstractmethod
    def fetch_data(self, query_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        دریافت داده‌های مورد نیاز بر اساس پارامترهای مشخص‌شده.

        :param query_params: پارامترهای جستجوی داده
        :return: داده‌های دریافت‌شده
        """
        pass

    @abstractmethod
    def store_data(self, data: Dict[str, Any]) -> None:
        """
        ذخیره داده‌های پردازش‌شده در پایگاه داده یا کش.

        :param data: داده‌های پردازش‌شده
        """
        pass
