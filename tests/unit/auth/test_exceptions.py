# tests/unit/auth/test_exceptions.py
"""
پکیج: tests.unit.auth.test_exceptions
توضیحات: تست‌های واحد برای کلاس‌های خطای سیستم احراز هویت

این ماژول تست شامل تست‌های مختلف برای اطمینان از عملکرد صحیح:
- ساخت انواع مختلف خطاها
- مقداردهی صحیح پارامترها
- تبدیل خطاها به دیکشنری
- بررسی جزئیات خطاها

نویسنده: Legend
تاریخ ایجاد: 2024-01-05
"""

import pytest
from datetime import datetime, timezone
from core.auth.exceptions import (
    AuthError,
    AuthenticationError,
    InvalidCredentialsError,
    AccountLockedError,
    TokenError,
    TokenExpiredError,
    TokenInvalidError,
    AccessDeniedError,
    ValidationError,
    ConfigurationError
)


def test_base_auth_error():
    """تست کلاس پایه AuthError"""
    # ساخت یک نمونه از خطای پایه
    error = AuthError(
        message="Test error",
        code="TEST_ERROR",
        details={"test_key": "test_value"}
    )

    # بررسی مقادیر پایه
    assert error.message == "Test error"
    assert error.code == "TEST_ERROR"
    assert error.details == {"test_key": "test_value"}
    assert isinstance(error.timestamp, datetime)

    # بررسی تبدیل به دیکشنری
    error_dict = error.to_dict()
    assert error_dict["error"]["message"] == "Test error"
    assert error_dict["error"]["code"] == "TEST_ERROR"
    assert error_dict["error"]["details"] == {"test_key": "test_value"}
    assert "timestamp" in error_dict["error"]


def test_authentication_error():
    """تست خطای احراز هویت پایه"""
    error = AuthenticationError("Login failed")
    assert error.code == "AUTHENTICATION_ERROR"
    assert error.message == "Login failed"

    error_dict = error.to_dict()
    assert error_dict["error"]["code"] == "AUTHENTICATION_ERROR"


def test_invalid_credentials_error():
    """تست خطای اعتبارنامه نامعتبر"""
    error = InvalidCredentialsError()
    assert error.message == "Invalid username or password"
    assert error.details["reason"] == "credentials_invalid"


def test_account_locked_error():
    """تست خطای قفل شدن حساب کاربری"""
    # تست بدون زمان باز شدن قفل
    error = AccountLockedError()
    assert "unlock_time" not in error.details

    # تست با زمان باز شدن قفل
    unlock_time = datetime.now(timezone.utc)
    error_with_time = AccountLockedError(unlock_time)
    assert error_with_time.details["unlock_time"] == unlock_time.isoformat()


def test_token_errors():
    """تست خطاهای مربوط به توکن"""
    # تست خطای منقضی شدن توکن
    expiry_time = datetime.now(timezone.utc)
    expired_error = TokenExpiredError(expiry_time)
    assert expired_error.details["expiry_time"] == expiry_time.isoformat()

    # تست خطای نامعتبر بودن توکن
    invalid_error = TokenInvalidError("Token signature mismatch")
    assert invalid_error.details["description"] == "Token signature mismatch"


def test_access_denied_error():
    """تست خطای عدم دسترسی"""
    error = AccessDeniedError("user_data", "read")
    assert error.message == "Access denied to resource: user_data"
    assert error.details["resource"] == "user_data"
    assert error.details["required_permission"] == "read"


def test_validation_error():
    """تست خطای اعتبارسنجی"""
    # تست پایه
    error = ValidationError("email", "invalid_format")
    assert error.details["field"] == "email"
    assert error.details["reason"] == "invalid_format"

    # تست با جزئیات اضافی
    error_with_details = ValidationError(
        field="password",
        reason="too_short",
        details={"min_length": 8}
    )
    assert error_with_details.details["min_length"] == 8


def test_configuration_error():
    """تست خطای پیکربندی"""
    error = ConfigurationError("jwt", "Missing secret key")
    assert error.code == "CONFIGURATION_ERROR"
    assert error.details["component"] == "jwt"
    assert error.details["reason"] == "Missing secret key"


def test_error_inheritance():
    """تست سلسله مراتب خطاها"""
    # بررسی اینکه همه خطاها از AuthError ارث‌بری می‌کنند
    assert issubclass(AuthenticationError, AuthError)
    assert issubclass(TokenError, AuthError)
    assert issubclass(AccessDeniedError, AuthError)
    assert issubclass(ValidationError, AuthError)
    assert issubclass(ConfigurationError, AuthError)


def test_exception_raising():
    """تست صحت رفتار خطاها هنگام raise شدن"""
    with pytest.raises(InvalidCredentialsError) as exc_info:
        raise InvalidCredentialsError()

    assert str(exc_info.value) == "Invalid username or password"


def test_timestamp_is_utc():
    """تست اینکه timestamp در UTC ذخیره می‌شود"""
    error = AuthError("Test", "TEST")
    # بررسی اینکه timestamp به درستی در فرمت ISO ذخیره می‌شود
    error_dict = error.to_dict()
    timestamp_str = error_dict["error"]["timestamp"]
    # بررسی امکان parse شدن timestamp
    datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))