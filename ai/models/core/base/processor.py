from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import time
import asyncio

class BaseProcessor(ABC):
    """
    کلاس پایه برای پردازش مدل‌ها.
    این کلاس یک ساختار استاندارد برای پردازش داده‌ها، مدیریت منابع و اجرای فرآیندهای مدل‌ها فراهم می‌کند.
    """

    def __init__(self, processor_name: str, config: Optional[Dict[str, Any]] = None):
        self.processor_name = processor_name
        self.config = config if config else {}
        self.status = "initialized"
        self.last_execution_time = None
        self.processing_time = 0.0

    @abstractmethod
    async def process(self, input_data: Any) -> Any:
        """
        متد پردازش داده که باید در کلاس‌های فرزند پیاده‌سازی شود.
        """
        pass

    async def execute(self, input_data: Any) -> Dict[str, Any]:
        """
        اجرای پردازش به‌صورت ناهمزمان و جمع‌آوری متریک‌های پردازشی.
        """
        self.status = "processing"
        start_time = time.time()

        try:
            result = await self.process(input_data)
            self.status = "completed"
        except Exception as e:
            self.status = "failed"
            result = {"error": str(e)}

        self.last_execution_time = time.time()
        self.processing_time = self.last_execution_time - start_time

        return {
            "processor_name": self.processor_name,
            "status": self.status,
            "execution_time": self.processing_time,
            "result": result
        }

    def update_config(self, new_config: Dict[str, Any]):
        """
        به‌روزرسانی تنظیمات پردازشگر.
        """
        self.config.update(new_config)

    def get_status(self) -> Dict[str, Any]:
        """
        دریافت وضعیت فعلی پردازشگر.
        """
        return {
            "processor_name": self.processor_name,
            "status": self.status,
            "last_execution_time": self.last_execution_time,
            "processing_time": self.processing_time
        }
