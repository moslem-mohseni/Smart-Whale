# infrastructure/clickhouse/security/audit_log.py
import os
import json
import logging
from loguru import logger as loguru_logger
from datetime import datetime
from typing import Dict, Any, Optional
from ..config.config import config

# تنظیم لاگر استاندارد پایتون
std_logger = logging.getLogger(__name__)


class AuditLogger:
    """
    سیستم ثبت لاگ‌های امنیتی و فعالیت‌های کاربران

    این کلاس از Loguru برای ثبت لاگ‌های امنیتی با فرمت JSON استفاده می‌کند.
    لاگ‌ها به صورت خودکار چرخش داده می‌شوند و مدت زمان نگهداری آنها قابل تنظیم است.
    """

    def __init__(self, log_dir: Optional[str] = None, app_name: str = "clickhouse"):
        """
        مقداردهی اولیه لاگر و تنظیم مسیر ذخیره لاگ‌ها

        Args:
            log_dir (str, optional): مسیر ذخیره‌سازی لاگ‌ها. اگر مشخص نشده باشد، از تنظیمات مرکزی استفاده می‌شود.
            app_name (str): نام برنامه برای درج در لاگ‌ها
        """
        # استفاده از مسیر ارسالی یا مسیر موجود در تنظیمات مرکزی
        self.log_dir = log_dir or os.path.dirname(config.get_data_management_config()["backup_dir"])
        if not self.log_dir:
            self.log_dir = "logs"  # مسیر پیش‌فرض

        self.app_name = app_name
        self.audit_log_path = os.path.join(self.log_dir, "audit")

        # ایجاد دایرکتوری لاگ اگر وجود نداشته باشد
        os.makedirs(self.audit_log_path, exist_ok=True)

        # تنظیم فایل لاگ اصلی
        self.audit_log_file = os.path.join(self.audit_log_path, f"{app_name}_audit.log")

        # تنظیم Loguru
        loguru_logger.remove()  # حذف همه handlers
        loguru_logger.add(
            self.audit_log_file,
            rotation="10 MB",  # چرخش خودکار هر 10 مگابایت
            compression="zip",  # فشرده‌سازی لاگ‌های قدیمی
            retention="90 days",  # نگهداری لاگ‌ها به مدت 90 روز
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}",
            serialize=True,  # ذخیره به فرمت JSON
            backtrace=True,  # ثبت traceback کامل در صورت خطا
            diagnose=True,  # ثبت جزئیات تشخیصی بیشتر
        )

        std_logger.info(f"Audit logger initialized. Logs stored at: {self.audit_log_file}")

    def log_event(self, username: str, action: str, status: str, details: str = "",
                  source_ip: Optional[str] = None, resource: Optional[str] = None):
        """
        ثبت یک رخداد امنیتی

        Args:
            username (str): نام کاربری
            action (str): عملیات انجام شده
            status (str): موفق یا ناموفق بودن عملیات
            details (str, optional): جزئیات اضافی درباره رخداد
            source_ip (str, optional): آدرس IP منبع درخواست
            resource (str, optional): منبع مورد دسترسی
        """
        log_message = {
            "timestamp": datetime.utcnow().isoformat(),
            "app": self.app_name,
            "username": username,
            "action": action,
            "status": status,
            "details": details,
            "source_ip": source_ip,
            "resource": resource
        }

        # حذف فیلدهای خالی
        log_message = {k: v for k, v in log_message.items() if v is not None}

        # ثبت لاگ
        loguru_logger.info(json.dumps(log_message))

    def log_security_event(self, event_type: str, username: str, success: bool, details: Dict[str, Any]):
        """
        ثبت یک رخداد امنیتی با جزئیات بیشتر

        Args:
            event_type (str): نوع رخداد امنیتی (مثلاً login, access_denied, permission_change)
            username (str): نام کاربری
            success (bool): آیا عملیات موفق بوده است
            details (Dict[str, Any]): جزئیات بیشتر درباره رخداد
        """
        status = "success" if success else "failure"

        # تبدیل details به رشته JSON اگر یک دیکشنری باشد
        if isinstance(details, dict):
            details_str = json.dumps(details)
        else:
            details_str = str(details)

        self.log_event(
            username=username,
            action=event_type,
            status=status,
            details=details_str
        )

    def get_audit_log_path(self) -> str:
        """
        دریافت مسیر فایل لاگ

        Returns:
            str: مسیر کامل فایل لاگ
        """
        return self.audit_log_file