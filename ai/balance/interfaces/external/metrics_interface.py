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


class KafkaInterface(ABC):
    """
    رابط Kafka برای ارتباط با پیام‌های پردازشی در ماژول Balance.
    """

    @abstractmethod
    def publish_message(self, topic: str, message: Dict[str, Any]) -> None:
        """
        انتشار پیام در یک تاپیک Kafka.

        :param topic: نام تاپیک
        :param message: داده‌های پیام
        """
        pass

    @abstractmethod
    def consume_message(self, topic: str) -> Dict[str, Any]:
        """
        دریافت پیام از یک تاپیک Kafka.

        :param topic: نام تاپیک
        :return: داده‌های پیام دریافت‌شده
        """
        pass


class MetricsInterface(ABC):
    """
    رابط متریک‌ها برای مدیریت و گزارش‌گیری داده‌های پردازشی در ماژول Balance.
    """

    @abstractmethod
    def collect_metrics(self) -> Dict[str, Any]:
        """
        جمع‌آوری متریک‌های عملکردی سیستم.

        :return: داده‌های متریک‌های جمع‌آوری‌شده
        """
        pass

    @abstractmethod
    def report_metrics(self, metrics_data: Dict[str, Any]) -> None:
        """
        گزارش متریک‌های پردازشی برای مانیتورینگ و تحلیل.

        :param metrics_data: داده‌های متریک برای گزارش‌دهی
        """
        pass
