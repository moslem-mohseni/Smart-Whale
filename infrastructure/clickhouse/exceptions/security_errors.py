# infrastructure/clickhouse/exceptions/security_errors.py
"""
خطاهای مرتبط با امنیت ClickHouse
"""

from .base import ClickHouseBaseError
from typing import Optional, Dict, Any


class SecurityError(ClickHouseBaseError):
    """
    خطای پایه برای مشکلات امنیتی ClickHouse
    """

    def __init__(self, message: str, code: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        """
        مقداردهی اولیه خطای امنیتی

        Args:
            message (str): پیام خطا
            code (str, optional): کد اختصاصی خطا
            details (Dict[str, Any], optional): جزئیات اضافی خطا
        """
        # تنظیم کد پیش‌فرض برای خطای امنیتی
        if code is None:
            code = "CHE300"

        super().__init__(message, code, details)


class EncryptionError(SecurityError):
    """
    خطای رمزنگاری یا رمزگشایی
    """

    def __init__(self, message: Optional[str] = None, code: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None, operation: Optional[str] = None):
        """
        مقداردهی اولیه خطای رمزنگاری

        Args:
            message (str, optional): پیام خطا
            code (str, optional): کد اختصاصی خطا
            details (Dict[str, Any], optional): جزئیات اضافی خطا
            operation (str, optional): عملیات مورد نظر ('encrypt' یا 'decrypt')
        """
        if message is None:
            if operation:
                message = f"Encryption error during {operation} operation"
            else:
                message = "Encryption error"

        if code is None:
            code = "CHE301"

        # اضافه کردن نوع عملیات به جزئیات
        if operation:
            if details is None:
                details = {}
            details["operation"] = operation

        super().__init__(message, code, details)


class TokenError(SecurityError):
    """
    خطای توکن JWT
    """

    def __init__(self, message: Optional[str] = None, code: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None, reason: Optional[str] = None):
        """
        مقداردهی اولیه خطای توکن

        Args:
            message (str, optional): پیام خطا
            code (str, optional): کد اختصاصی خطا
            details (Dict[str, Any], optional): جزئیات اضافی خطا
            reason (str, optional): دلیل خطای توکن
        """
        if message is None:
            if reason:
                message = f"Token error: {reason}"
            else:
                message = "Token error"

        if code is None:
            code = "CHE302"

        # اضافه کردن دلیل خطای توکن به جزئیات
        if reason:
            if details is None:
                details = {}
            details["reason"] = reason

        super().__init__(message, code, details)


class PermissionDeniedError(SecurityError):
    """
    خطای عدم دسترسی
    """

    def __init__(self, message: Optional[str] = None, code: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None, username: Optional[str] = None,
                 resource: Optional[str] = None, required_permission: Optional[str] = None):
        """
        مقداردهی اولیه خطای عدم دسترسی

        Args:
            message (str, optional): پیام خطا
            code (str, optional): کد اختصاصی خطا
            details (Dict[str, Any], optional): جزئیات اضافی خطا
            username (str, optional): نام کاربری
            resource (str, optional): منبع مورد نظر
            required_permission (str, optional): دسترسی مورد نیاز
        """
        if message is None:
            if resource and required_permission:
                message = f"Permission denied: '{required_permission}' required for resource '{resource}'"
            else:
                message = "Permission denied"

        if code is None:
            code = "CHE303"

        # اضافه کردن اطلاعات دسترسی به جزئیات
        if username or resource or required_permission:
            if details is None:
                details = {}
            if username:
                details["username"] = username
            if resource:
                details["resource"] = resource
            if required_permission:
                details["required_permission"] = required_permission

        super().__init__(message, code, details)
        