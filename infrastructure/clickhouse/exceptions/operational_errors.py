# infrastructure/clickhouse/exceptions/operational_errors.py
"""
خطاهای عملیاتی مرتبط با ClickHouse
"""

from .base import ClickHouseBaseError
from typing import Optional, Dict, Any


class OperationalError(ClickHouseBaseError):
    """
    خطای پایه برای مشکلات عملیاتی ClickHouse
    """

    def __init__(self, message: str, code: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        """
        مقداردهی اولیه خطای عملیاتی

        Args:
            message (str): پیام خطا
            code (str, optional): کد اختصاصی خطا
            details (Dict[str, Any], optional): جزئیات اضافی خطا
        """
        # تنظیم کد پیش‌فرض برای خطای عملیاتی
        if code is None:
            code = "CHE400"

        super().__init__(message, code, details)


class CircuitBreakerError(OperationalError):
    """
    خطای Circuit Breaker
    """

    def __init__(self, message: Optional[str] = None, code: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None, failure_count: Optional[int] = None):
        """
        مقداردهی اولیه خطای Circuit Breaker

        Args:
            message (str, optional): پیام خطا
            code (str, optional): کد اختصاصی خطا
            details (Dict[str, Any], optional): جزئیات اضافی خطا
            failure_count (int, optional): تعداد خطاهای متوالی
        """
        if message is None:
            message = "Circuit breaker is open. Requests are blocked."

        if code is None:
            code = "CHE401"

        # اضافه کردن تعداد خطاها به جزئیات
        if failure_count is not None:
            if details is None:
                details = {}
            details["failure_count"] = failure_count

        super().__init__(message, code, details)


class RetryExhaustedError(OperationalError):
    """
    خطای اتمام تلاش‌های مجدد
    """

    def __init__(self, message: Optional[str] = None, code: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None, attempts: Optional[int] = None,
                 last_error: Optional[str] = None):
        """
        مقداردهی اولیه خطای اتمام تلاش‌های مجدد

        Args:
            message (str, optional): پیام خطا
            code (str, optional): کد اختصاصی خطا
            details (Dict[str, Any], optional): جزئیات اضافی خطا
            attempts (int, optional): تعداد تلاش‌های انجام شده
            last_error (str, optional): آخرین خطای رخ داده
        """
        if message is None:
            if attempts:
                message = f"Max retry attempts ({attempts}) reached for ClickHouse operation."
            else:
                message = "Max retry attempts reached for ClickHouse operation."

        if code is None:
            code = "CHE402"

        # اضافه کردن اطلاعات تلاش‌ها به جزئیات
        if attempts is not None or last_error:
            if details is None:
                details = {}
            if attempts is not None:
                details["attempts"] = attempts
            if last_error:
                details["last_error"] = last_error

        super().__init__(message, code, details)


class BackupError(OperationalError):
    """
    خطای عملیات پشتیبان‌گیری یا بازیابی
    """

    def __init__(self, message: Optional[str] = None, code: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None, operation: Optional[str] = None,
                 table_name: Optional[str] = None, backup_file: Optional[str] = None):
        """
        مقداردهی اولیه خطای پشتیبان‌گیری

        Args:
            message (str, optional): پیام خطا
            code (str, optional): کد اختصاصی خطا
            details (Dict[str, Any], optional): جزئیات اضافی خطا
            operation (str, optional): نوع عملیات ('backup' یا 'restore')
            table_name (str, optional): نام جدول
            backup_file (str, optional): مسیر فایل پشتیبان
        """
        if message is None:
            if operation and table_name:
                message = f"Failed to {operation} table {table_name}"
            elif operation:
                message = f"Failed to {operation}"
            else:
                message = "Backup operation failed"

        if code is None:
            code = "CHE403"

        # اضافه کردن اطلاعات پشتیبان‌گیری به جزئیات
        if operation or table_name or backup_file:
            if details is None:
                details = {}
            if operation:
                details["operation"] = operation
            if table_name:
                details["table_name"] = table_name
            if backup_file:
                details["backup_file"] = backup_file

        super().__init__(message, code, details)


class DataManagementError(OperationalError):
    """
    خطای مدیریت داده
    """

    def __init__(self, message: Optional[str] = None, code: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None, operation: Optional[str] = None,
                 table_name: Optional[str] = None):
        """
        مقداردهی اولیه خطای مدیریت داده

        Args:
            message (str, optional): پیام خطا
            code (str, optional): کد اختصاصی خطا
            details (Dict[str, Any], optional): جزئیات اضافی خطا
            operation (str, optional): نوع عملیات (مثلاً 'optimize', 'delete_expired')
            table_name (str, optional): نام جدول
        """
        if message is None:
            if operation and table_name:
                message = f"Failed to {operation} data in table {table_name}"
            elif operation:
                message = f"Failed to {operation} data"
            else:
                message = "Data management operation failed"

        if code is None:
            code = "CHE404"

        # اضافه کردن اطلاعات عملیات به جزئیات
        if operation or table_name:
            if details is None:
                details = {}
            if operation:
                details["operation"] = operation
            if table_name:
                details["table_name"] = table_name

        super().__init__(message, code, details)
