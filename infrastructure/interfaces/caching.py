# infrastructure/interfaces/caching.py

from abc import ABC, abstractmethod
from typing import Any, Optional

class CachingInterface(ABC):
    """
    رابط پایه برای پیاده‌سازی‌های مختلف کش
    """

    @abstractmethod
    async def connect(self) -> None:
        """برقراری اتصال به سرور کش"""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """قطع اتصال از سرور کش"""
        pass

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """بازیابی مقدار از کش"""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """ذخیره مقدار در کش"""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """حذف کلید از کش"""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """بررسی وجود کلید در کش"""
        pass

    @abstractmethod
    async def ttl(self, key: str) -> Optional[int]:
        """دریافت زمان انقضای کلید"""
        pass

    @abstractmethod
    async def scan_keys(self, pattern: str) -> list:
        """جستجوی کلیدها با الگو"""
        pass