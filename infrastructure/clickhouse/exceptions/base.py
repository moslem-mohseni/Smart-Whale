# infrastructure/clickhouse/exceptions/base.py
"""
کلاس پایه خطاهای ClickHouse
"""

import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class ClickHouseBaseError(Exception):
    """
    کلاس پایه برای تمامی خطاهای سفارشی ClickHouse

    این کلاس پایه مشترک برای تمامی خطاهای سفارشی ClickHouse است و
    قابلیت‌های مشترک مانند لاگینگ و ذخیره اطلاعات اضافی را فراهم می‌کند.
    """

    def __init__(self, message: str, code: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        """
        مقداردهی اولیه خطای پایه

        Args:
            message (str): پیام خطا
            code (str, optional): کد اختصاصی خطا
            details (Dict[str, Any], optional): جزئیات اضافی خطا به صورت دیکشنری
        """
        self.message = message
        self.code = code or "CHE001"  # کد پیش‌فرض خطا
        self.details = details or {}

        # فراخوانی سازنده کلاس پایه
        super().__init__(self.message)

        # ثبت خطا در لاگ
        self._log_error()

    def _log_error(self):
        """
        ثبت خطا در لاگ
        """
        log_message = f"{self.__class__.__name__} [{self.code}]: {self.message}"

        if self.details:
            log_message += f" | Details: {self.details}"

        logger.error(log_message)

    def to_dict(self) -> Dict[str, Any]:
        """
        تبدیل خطا به دیکشنری برای استفاده در API

        Returns:
            Dict[str, Any]: دیکشنری شامل اطلاعات خطا
        """
        error_dict = {
            "error_type": self.__class__.__name__,
            "code": self.code,
            "message": self.message
        }

        if self.details:
            error_dict["details"] = self.details

        return error_dict

    def __str__(self) -> str:
        """
        تبدیل خطا به رشته برای نمایش

        Returns:
            str: نمایش رشته‌ای خطا
        """
        result = f"{self.__class__.__name__} [{self.code}]: {self.message}"

        if self.details:
            result += f" | Details: {self.details}"

        return result
    