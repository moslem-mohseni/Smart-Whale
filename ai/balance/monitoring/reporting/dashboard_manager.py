from abc import ABC, abstractmethod
from typing import Dict, Any

class DashboardManager(ABC):
    """
    مدیریت داشبوردهای نمایشی برای مانیتورینگ عملکرد سیستم در ماژول Balance.
    """

    @abstractmethod
    def update_dashboard(self, dashboard_data: Dict[str, Any]) -> None:
        """
        بروزرسانی داشبورد مانیتورینگ با داده‌های جدید.

        :param dashboard_data: داده‌های پردازشی برای نمایش در داشبورد
        """
        pass

    @abstractmethod
    def get_dashboard_snapshot(self) -> Dict[str, Any]:
        """
        دریافت نمایی از وضعیت فعلی داشبورد.

        :return: داده‌های فعلی داشبورد برای نمایش
        """
        pass
