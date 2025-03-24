# infrastructure/clickhouse/exceptions/connection_errors.py
"""
خطاهای مرتبط با اتصال به ClickHouse
"""

from .base import ClickHouseBaseError
from typing import Optional, Dict, Any


class ConnectionError(ClickHouseBaseError):
    """
    خطای پایه برای مشکلات اتصال به ClickHouse
    """

    def __init__(self, message: str, code: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None, host: Optional[str] = None):
        """
        مقداردهی اولیه خطای اتصال

        Args:
            message (str): پیام خطا
            code (str, optional): کد اختصاصی خطا
            details (Dict[str, Any], optional): جزئیات اضافی خطا
            host (str, optional): هاست مورد نظر
        """
        # تنظیم کد پیش‌فرض برای خطای اتصال
        if code is None:
            code = "CHE100"

        # اضافه کردن اطلاعات هاست به جزئیات
        if host:
            if details is None:
                details = {}
            details["host"] = host

        super().__init__(message, code, details)


class PoolExhaustedError(ConnectionError):
    """
    خطای اتمام ظرفیت Connection Pool
    """

    def __init__(self, message: Optional[str] = None, code: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None, max_connections: Optional[int] = None):
        """
        مقداردهی اولیه خطای اتمام ظرفیت

        Args:
            message (str, optional): پیام خطا
            code (str, optional): کد اختصاصی خطا
            details (Dict[str, Any], optional): جزئیات اضافی خطا
            max_connections (int, optional): حداکثر تعداد اتصالات مجاز
        """
        if message is None:
            message = "Connection pool exhausted. No available connections."

        if code is None:
            code = "CHE101"

        # اضافه کردن اطلاعات حداکثر اتصالات به جزئیات
        if max_connections:
            if details is None:
                details = {}
            details["max_connections"] = max_connections

        super().__init__(message, code, details)


class ConnectionTimeoutError(ConnectionError):
    """
    خطای زمان انتظار برای اتصال به ClickHouse
    """

    def __init__(self, message: Optional[str] = None, code: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None, host: Optional[str] = None,
                 timeout: Optional[float] = None):
        """
        مقداردهی اولیه خطای زمان انتظار

        Args:
            message (str, optional): پیام خطا
            code (str, optional): کد اختصاصی خطا
            details (Dict[str, Any], optional): جزئیات اضافی خطا
            host (str, optional): هاست مورد نظر
            timeout (float, optional): مدت زمان انتظار (ثانیه)
        """
        if message is None:
            message = "Connection timed out while connecting to ClickHouse"

        if code is None:
            code = "CHE102"

        # اضافه کردن اطلاعات زمان انتظار به جزئیات
        if timeout:
            if details is None:
                details = {}
            details["timeout"] = timeout

        super().__init__(message, code, details, host)


class AuthenticationError(ConnectionError):
    """
    خطای احراز هویت در اتصال به ClickHouse
    """

    def __init__(self, message: Optional[str] = None, code: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None, host: Optional[str] = None,
                 user: Optional[str] = None):
        """
        مقداردهی اولیه خطای احراز هویت

        Args:
            message (str, optional): پیام خطا
            code (str, optional): کد اختصاصی خطا
            details (Dict[str, Any], optional): جزئیات اضافی خطا
            host (str, optional): هاست مورد نظر
            user (str, optional): نام کاربری
        """
        if message is None:
            message = "Authentication failed while connecting to ClickHouse"

        if code is None:
            code = "CHE103"

        # اضافه کردن اطلاعات کاربر به جزئیات
        if user:
            if details is None:
                details = {}
            details["user"] = user

        super().__init__(message, code, details, host)
        