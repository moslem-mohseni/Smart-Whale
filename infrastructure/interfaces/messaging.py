# infrastructure/interfaces/messaging.py

from abc import ABC, abstractmethod
from typing import Any, Callable, Awaitable


class MessagingInterface(ABC):
    """
    اینترفیس پایه برای سیستم‌های پیام‌رسانی

    این اینترفیس عملیات اصلی مورد نیاز برای ارتباط بین سرویس‌ها را تعریف می‌کند.
    برای مثال، Kafka باید این اینترفیس را پیاده‌سازی کند.
    """

    @abstractmethod
    async def connect(self) -> None:
        """
        برقراری اتصال به سرور پیام‌رسان

        این متد باید اتصال اولیه به سرور پیام‌رسان را برقرار کند و
        در صورت بروز مشکل، خطای مناسب را صادر کند.
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """
        قطع اتصال از سرور پیام‌رسان

        این متد باید تمام منابع و اتصالات را به درستی آزاد کند.
        """
        pass

    @abstractmethod
    async def publish(self, topic: str, message: Any) -> None:
        """
        انتشار یک پیام در یک موضوع مشخص

        Args:
            topic: نام موضوع
            message: پیامی که باید منتشر شود
        """
        pass

    @abstractmethod
    async def subscribe(self, topic: str, handler: Callable[[Any], Awaitable[None]]) -> None:
        """
        اشتراک در یک موضوع برای دریافت پیام‌ها

        Args:
            topic: نام موضوع
            handler: تابعی که برای پردازش پیام‌های دریافتی فراخوانی می‌شود
        """
        pass

    @abstractmethod
    async def unsubscribe(self, topic: str) -> None:
        """
        لغو اشتراک از یک موضوع

        Args:
            topic: نام موضوع
        """
        pass