from typing import Dict, Any

class MemoryAllocator:
    """
    تخصیص حافظه به مدل‌ها و مدیریت استفاده بهینه از حافظه.
    """

    def __init__(self, max_memory: int = 8192):
        """
        مقداردهی اولیه مدیریت حافظه.
        :param max_memory: حداکثر مقدار حافظه در مگابایت.
        """
        self.max_memory = max_memory  # کل حافظه موجود
        self.allocated_memory: Dict[str, int] = {}  # حافظه تخصیص‌یافته به مدل‌ها

    def allocate(self, model_id: str, size: int) -> bool:
        """
        تخصیص حافظه به یک مدل خاص.
        :param model_id: شناسه مدل.
        :param size: مقدار حافظه مورد نیاز (MB).
        :return: موفقیت یا شکست تخصیص حافظه.
        """
        if self.get_available_memory() >= size:
            self.allocated_memory[model_id] = size
            return True
        return False  # تخصیص ناموفق به دلیل کمبود حافظه

    def release(self, model_id: str):
        """
        آزادسازی حافظه اختصاص‌یافته به مدل.
        :param model_id: شناسه مدل.
        """
        if model_id in self.allocated_memory:
            del self.allocated_memory[model_id]

    def get_available_memory(self) -> int:
        """
        دریافت میزان حافظه در دسترس.
        :return: مقدار حافظه باقی‌مانده (MB).
        """
        return self.max_memory - sum(self.allocated_memory.values())

    def get_status(self) -> Dict[str, Any]:
        """
        دریافت وضعیت تخصیص حافظه.
        :return: دیکشنری شامل اطلاعات حافظه.
        """
        return {
            "total_memory": self.max_memory,
            "allocated_memory": self.allocated_memory,
            "available_memory": self.get_available_memory()
        }
