# infrastructure/interfaces/storage.py

from abc import ABC, abstractmethod
from typing import List, Any, Optional

class StorageInterface(ABC):
    """
    اینترفیس پایه برای ذخیره‌سازی داده

    این اینترفیس عملیات پایه برای کار با پایگاه داده را تعریف می‌کند.
    هر پیاده‌سازی جدید برای ذخیره‌سازی باید این اینترفیس را پیاده‌سازی کند.
    """

    @abstractmethod
    async def connect(self) -> None:
        """برقراری اتصال به پایگاه داده"""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """قطع اتصال از پایگاه داده"""
        pass

    @abstractmethod
    async def is_connected(self) -> bool:
        """بررسی وضعیت اتصال"""
        pass

    @abstractmethod
    async def execute(self, query: str, params: Optional[List[Any]] = None) -> List[Any]:
        """
        اجرای یک پرس‌وجو

        Args:
            query: پرس‌وجوی SQL
            params: پارامترهای پرس‌وجو (اختیاری)

        Returns:
            نتیجه پرس‌وجو
        """
        pass

    @abstractmethod
    async def execute_many(self, query: str, params_list: List[List[Any]]) -> None:
        """
        اجرای یک پرس‌وجو با چندین سری پارامتر

        Args:
            query: پرس‌وجوی SQL
            params_list: لیست پارامترها
        """
        pass

    @abstractmethod
    async def begin_transaction(self) -> None:
        """شروع یک تراکنش"""
        pass

    @abstractmethod
    async def commit(self) -> None:
        """تایید تراکنش"""
        pass

    @abstractmethod
    async def rollback(self) -> None:
        """برگشت تراکنش"""
        pass

    @abstractmethod
    async def create_table(self, table_name: str, schema: dict) -> None:
        """
        ایجاد جدول جدید

        Args:
            table_name: نام جدول
            schema: ساختار جدول
        """
        pass

    @abstractmethod
    async def create_hypertable(self, table_name: str, time_column: str) -> None:
        """
        تبدیل یک جدول به hypertable

        Args:
            table_name: نام جدول
            time_column: نام ستون زمان
        """
        pass