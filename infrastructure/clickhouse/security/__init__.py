# infrastructure/clickhouse/security/__init__.py
"""
ماژول امنیت ClickHouse

این ماژول شامل کلاس‌ها و توابع مربوط به امنیت و کنترل دسترسی در ClickHouse است:
- AccessControl: مدیریت توکن‌های JWT و سیستم کنترل دسترسی
- AuditLogger: ثبت رخدادهای امنیتی و فعالیت‌های کاربران
- EncryptionManager: رمزنگاری و رمزگشایی داده‌های حساس
"""

import logging
from .access_control import AccessControl
from .audit_log import AuditLogger
from .encryption import EncryptionManager

logger = logging.getLogger(__name__)

logger.info("Initializing ClickHouse Security Module...")

__all__ = [
    "AccessControl",
    "AuditLogger",
    "EncryptionManager",
    "create_encryption_manager",
    "create_access_control",
    "create_audit_logger"
]


def create_encryption_manager(key=None):
    """
    ایجاد یک نمونه از EncryptionManager با تنظیمات مناسب

    Args:
        key (str, optional): کلید رمزنگاری سفارشی

    Returns:
        EncryptionManager: نمونه آماده استفاده از EncryptionManager
    """
    return EncryptionManager(key=key)


def create_access_control(secret_key=None, token_expiry=None):
    """
    ایجاد یک نمونه از AccessControl با تنظیمات مناسب

    Args:
        secret_key (str, optional): کلید محرمانه سفارشی
        token_expiry (int, optional): مدت اعتبار توکن (ثانیه)

    Returns:
        AccessControl: نمونه آماده استفاده از AccessControl
    """
    return AccessControl(secret_key=secret_key, token_expiry=token_expiry)


def create_audit_logger(log_dir=None, app_name="clickhouse"):
    """
    ایجاد یک نمونه از AuditLogger با تنظیمات مناسب

    Args:
        log_dir (str, optional): مسیر سفارشی ذخیره‌سازی لاگ‌ها
        app_name (str, optional): نام برنامه

    Returns:
        AuditLogger: نمونه آماده استفاده از AuditLogger
    """
    return AuditLogger(log_dir=log_dir, app_name=app_name)
