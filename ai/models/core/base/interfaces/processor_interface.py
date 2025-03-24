from abc import ABC, abstractmethod
from typing import Any, Dict

class ProcessorInterface(ABC):
    """
    اینترفیس استاندارد برای پردازشگرهای مدل‌ها.
    """

    @abstractmethod
    async def process(self, input_data: Any) -> Any:
        """
        اجرای پردازش روی داده‌ی ورودی و برگرداندن نتیجه.
        """
        pass

    @abstractmethod
    async def execute(self, input_data: Any) -> Dict[str, Any]:
        """
        اجرای پردازش همراه با نظارت بر زمان پردازش و وضعیت.
        """
        pass

    @abstractmethod
    def update_config(self, new_config: Dict[str, Any]):
        """
        بروزرسانی تنظیمات پردازشگر.
        """
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """
        دریافت وضعیت پردازشگر.
        """
        pass
