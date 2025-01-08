# core/auth/exceptions.py
"""
پکیج: core.auth.exceptions
توضیحات: کلاس‌های خطای اختصاصی برای سیستم احراز هویت و کنترل دسترسی
این ماژول خطاهای مختلف مرتبط با:
- احراز هویت کاربران
- مدیریت توکن‌ها
- کنترل دسترسی‌ها
- اعتبارسنجی داده‌ها
را تعریف می‌کند.

نویسنده: Legend
تاریخ ایجاد: 2024-01-05
"""

from typing import Optional, Any, Dict
from datetime import datetime


class AuthError(Exception):
    """کلاس پایه برای تمام خطاهای مربوط به احراز هویت و دسترسی"""

    def __init__(self, message: str, code: str, details: Optional[Dict[str, Any]] = None):
        """
        راه‌اندازی خطای احراز هویت

        Args:
            message: پیام خطا برای نمایش به کاربر
            code: کد یکتای خطا برای شناسایی نوع خطا
            details: جزئیات اضافی خطا (اختیاری)
        """
        self.message = message
        self.code = code
        self.details = details or {}
        self.timestamp = datetime.utcnow()
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """تبدیل خطا به دیکشنری برای ارسال در API"""
        return {
            'error': {
                'code': self.code,
                'message': self.message,
                'details': self.details,
                'timestamp': self.timestamp.isoformat()
            }
        }


class AuthenticationError(AuthError):
    """خطاهای مربوط به فرآیند احراز هویت"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            code='AUTHENTICATION_ERROR',
            details=details
        )


class InvalidCredentialsError(AuthenticationError):
    """خطای نام کاربری یا رمز عبور نادرست"""

    def __init__(self):
        super().__init__(
            message="Invalid username or password",
            details={'reason': 'credentials_invalid'}
        )


class AccountLockedError(AuthenticationError):
    """خطای قفل شدن حساب کاربری"""

    def __init__(self, unlock_time: Optional[datetime] = None):
        details = {'reason': 'account_locked'}
        if unlock_time:
            details['unlock_time'] = unlock_time.isoformat()

        super().__init__(
            message="Account is locked. Please try again later",
            details=details
        )


class TokenError(AuthError):
    """خطاهای مربوط به توکن‌های احراز هویت"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            code='TOKEN_ERROR',
            details=details
        )


class TokenExpiredError(TokenError):
    """خطای منقضی شدن توکن"""

    def __init__(self, expiry_time: datetime):
        super().__init__(
            message="Token has expired",
            details={
                'reason': 'token_expired',
                'expiry_time': expiry_time.isoformat()
            }
        )


class TokenInvalidError(TokenError):
    """خطای نامعتبر بودن توکن"""

    def __init__(self, reason: str):
        super().__init__(
            message="Invalid token",
            details={
                'reason': 'token_invalid',
                'description': reason
            }
        )


class AccessDeniedError(AuthError):
    """خطای عدم دسترسی به منبع"""

    def __init__(self, resource: str, required_permission: str):
        super().__init__(
            message=f"Access denied to resource: {resource}",
            code='ACCESS_DENIED',
            details={
                'resource': resource,
                'required_permission': required_permission
            }
        )


class ValidationError(AuthError):
    """خطاهای مربوط به اعتبارسنجی داده‌های ورودی"""

    def __init__(self, field: str, reason: str, details: Optional[Dict[str, Any]] = None):
        error_details = {
            'field': field,
            'reason': reason
        }
        if details:
            error_details.update(details)

        super().__init__(
            message=f"Validation error for field: {field}",
            code='VALIDATION_ERROR',
            details=error_details
        )


class ConfigurationError(AuthError):
    """خطاهای مربوط به پیکربندی نادرست سیستم احراز هویت"""

    def __init__(self, component: str, reason: str):
        super().__init__(
            message=f"Authentication system configuration error in {component}",
            code='CONFIGURATION_ERROR',
            details={
                'component': component,
                'reason': reason
            }
        )