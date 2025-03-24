from abc import ABC, abstractmethod
from typing import Any, Dict

class ModelInterface(ABC):
    """
    اینترفیس استاندارد برای تمامی مدل‌های هوش مصنوعی.
    """

    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """
        پردازش داده ورودی و تولید خروجی. این متد باید در هر مدل پیاده‌سازی شود.
        """
        pass

    @abstractmethod
    def save_state(self) -> str:
        """
        ذخیره وضعیت مدل در قالب JSON.
        """
        pass

    @abstractmethod
    def load_state(self, state_json: str):
        """
        بارگذاری وضعیت مدل از JSON.
        """
        pass

    @abstractmethod
    def update_config(self, new_config: Dict[str, Any]):
        """
        بروزرسانی تنظیمات مدل.
        """
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """
        دریافت وضعیت فعلی مدل.
        """
        pass
