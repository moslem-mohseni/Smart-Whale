"""
NeedDetectorBase Module
------------------------
این فایل کلاس پایه (abstract base) برای تشخیص نیازهای یادگیری در سیستم Self-Learning را تعریف می‌کند.
این کلاس پایه به سایر زیرکلاس‌ها اجازه می‌دهد که نیازهای یادگیری را بر اساس داده‌های ورودی مختلف
(مانند متریک‌های عملکرد، بازخورد کاربر، تحلیل شکاف دانشی و ...) شناسایی و دسته‌بندی کنند.

نسخه نهایی و عملیاتی با بهترین مکانیسم‌های هوشمند و کارایی بالا.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List


class NeedDetectorBase(ABC):
    """
    کلاس پایه برای تشخیص نیازهای یادگیری (Learning Needs).
    زیرکلاس‌ها باید متد `detect_needs` را پیاده‌سازی کنند تا نیازهای یادگیری
    بر اساس داده‌های ورودی خاص شناسایی شود.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        راه‌اندازی اولیه تشخیص‌دهنده نیاز.

        Args:
            config (Optional[Dict[str, Any]]): پیکربندی اختصاصی برای تشخیص نیاز.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config or {}
        self.logger.info(f"[{self.__class__.__name__}] Initialized with config: {self.config}")

    @abstractmethod
    def detect_needs(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        متد انتزاعی برای شناسایی نیازهای یادگیری بر اساس داده‌های ورودی.

        Args:
            input_data (Dict[str, Any]): داده‌های لازم برای تحلیل نیازها (مثل متریک‌های عملکرد، گزارش بازخوردها، شکاف‌های دانشی و ...).

        Returns:
            List[Dict[str, Any]]: لیستی از نیازهای یادگیری شناسایی‌شده؛
                                  هر آیتم می‌تواند شامل نوع نیاز، سطح اهمیت و سایر جزئیات باشد.
        """
        pass

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        اعتبارسنجی داده‌های ورودی به صورت پایه.
        زیرکلاس‌ها می‌توانند این متد را بازنویسی کنند تا قوانین اختصاصی داشته باشند.

        Args:
            input_data (Dict[str, Any]): داده‌های ورودی

        Returns:
            bool: نتیجه اعتبارسنجی (True در صورت معتبر بودن)
        """
        if not input_data:
            self.logger.warning(f"[{self.__class__.__name__}] Received empty input data.")
            return False
        return True

    def post_process_needs(self, detected_needs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        متد پایه برای پردازش نهایی نیازهای شناسایی‌شده.
        می‌توان این متد را در زیرکلاس‌ها override کرد تا مثلاً نیازهای مشابه ادغام شوند
        یا به ترتیب اهمیت مرتب گردند.

        Args:
            detected_needs (List[Dict[str, Any]]): لیست نیازهای شناسایی‌شده.

        Returns:
            List[Dict[str, Any]]: لیست نهایی نیازهای اصلاح‌شده.
        """
        # در این نسخه پایه، بدون تغییر خاصی خروجی را برمی‌گردانیم.
        return detected_needs

    def get_config_param(self, param_name: str, default: Any = None) -> Any:
        """
        متد کمکی برای دسترسی به پارامترهای پیکربندی.

        Args:
            param_name (str): نام پارامتر پیکربندی
            default (Any): مقدار پیش‌فرض در صورت عدم وجود پارامتر

        Returns:
            Any: مقدار پارامتر یا مقدار پیش‌فرض
        """
        return self.config.get(param_name, default)
