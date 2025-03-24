from abc import ABC, abstractmethod
from typing import Dict, Any

class ResourceAware(ABC):
    """
    اینترفیس استاندارد برای مدیریت منابع مدل‌ها و پردازشگرها.
    """

    @abstractmethod
    def allocate_resources(self, cpu: float, memory: float, gpu: float = 0.0):
        """
        تخصیص منابع پردازشی (CPU، حافظه، و در صورت نیاز GPU).
        """
        pass

    @abstractmethod
    def release_resources(self):
        """
        آزادسازی منابع پس از اتمام پردازش.
        """
        pass

    @abstractmethod
    def get_resource_status(self) -> Dict[str, Any]:
        """
        دریافت وضعیت فعلی منابع تخصیص‌یافته.
        """
        pass
