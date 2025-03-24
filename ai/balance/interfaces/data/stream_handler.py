from abc import ABC, abstractmethod
from typing import Any, Dict, Generator

class StreamHandler(ABC):
    """
    مدیریت جریان داده‌ها برای پردازش و انتقال داده‌ها در ماژول Balance.
    """

    @abstractmethod
    def open_stream(self, query_params: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
        """
        ایجاد و مدیریت یک جریان داده برای خواندن پیوسته داده‌ها.

        :param query_params: پارامترهای جستجوی داده
        :yield: داده‌های دریافت‌شده به‌صورت جریانی
        """
        pass

    @abstractmethod
    def close_stream(self) -> None:
        """
        بستن جریان داده برای جلوگیری از نشت منابع.
        """
        pass
