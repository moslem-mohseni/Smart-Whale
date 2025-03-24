from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import json
import time

class BaseModel(ABC):
    """
    کلاس پایه برای تمامی مدل‌های هوش مصنوعی.
    این کلاس شامل متدهای پایه‌ای برای مدیریت پردازش، حافظه، منابع و یادگیری فدراسیونی است.
    """

    def __init__(self, model_name: str, version: str, config: Optional[Dict[str, Any]] = None):
        self.model_name = model_name
        self.version = version
        self.config = config if config else {}
        self.status = "initialized"
        self.last_updated = time.time()

    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """
        متد پردازش ورودی که باید در کلاس‌های فرزند پیاده‌سازی شود.
        """
        pass

    def save_state(self) -> str:
        """
        ذخیره وضعیت مدل در قالب JSON.
        """
        state = {
            "model_name": self.model_name,
            "version": self.version,
            "config": self.config,
            "status": self.status,
            "last_updated": self.last_updated
        }
        return json.dumps(state)

    def load_state(self, state_json: str):
        """
        بارگذاری وضعیت مدل از JSON.
        """
        state = json.loads(state_json)
        self.model_name = state.get("model_name", self.model_name)
        self.version = state.get("version", self.version)
        self.config = state.get("config", self.config)
        self.status = state.get("status", self.status)
        self.last_updated = state.get("last_updated", self.last_updated)

    def update_config(self, new_config: Dict[str, Any]):
        """
        به‌روزرسانی تنظیمات مدل و اعمال تغییرات لازم.
        """
        self.config.update(new_config)
        self.last_updated = time.time()

    def get_status(self) -> Dict[str, Any]:
        """
        دریافت وضعیت فعلی مدل.
        """
        return {
            "model_name": self.model_name,
            "version": self.version,
            "status": self.status,
            "last_updated": self.last_updated
        }
