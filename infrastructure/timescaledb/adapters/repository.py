# infrastructure/timescaledb/adapters/repository.py

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional, List
from datetime import datetime

T = TypeVar('T')


class Repository(ABC, Generic[T]):
    """
    اینترفیس پایه برای دسترسی به داده‌ها

    این کلاس انتزاعی الگوی Repository را پیاده‌سازی می‌کند.
    هر نوع داده‌ای که نیاز به ذخیره‌سازی دارد باید repository خاص خود را داشته باشد.
    """

    @abstractmethod
    async def add(self, entity: T) -> T:
        """افزودن یک موجودیت جدید"""
        pass

    @abstractmethod
    async def get(self, id: str) -> Optional[T]:
        """بازیابی یک موجودیت با شناسه"""
        pass

    @abstractmethod
    async def update(self, entity: T) -> Optional[T]:
        """به‌روزرسانی یک موجودیت"""
        pass

    @abstractmethod
    async def delete(self, id: str) -> bool:
        """حذف یک موجودیت"""
        pass
