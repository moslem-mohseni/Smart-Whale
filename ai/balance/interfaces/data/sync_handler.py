from abc import ABC, abstractmethod
from typing import Any, Dict

class SyncHandler(ABC):
    """
    مدیریت همگام‌سازی داده‌ها بین اجزای مختلف ماژول Balance.
    """

    @abstractmethod
    def sync_data(self, source: str, destination: str, data: Dict[str, Any]) -> None:
        """
        همگام‌سازی داده‌ها بین دو منبع مشخص.

        :param source: منبع داده
        :param destination: مقصد داده
        :param data: داده‌هایی که باید همگام‌سازی شوند
        """
        pass

    @abstractmethod
    def verify_sync(self, source: str, destination: str) -> bool:
        """
        بررسی صحت فرآیند همگام‌سازی داده‌ها.

        :param source: منبع داده
        :param destination: مقصد داده
        :return: نتیجه بررسی همگام‌سازی (True/False)
        """
        pass
