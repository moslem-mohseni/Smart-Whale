import logging
from typing import Callable, Any


class FallbackManager:
    def __init__(self):
        """
        مدیریت مکانیزم‌های جایگزین (Fallback) برای جلوگیری از خرابی کامل سیستم
        """
        self.logger = logging.getLogger("FallbackManager")

    def execute_with_fallback(self, function: Callable, fallback_value: Any = None, fallback_function: Callable = None, *args, **kwargs):
        """
        اجرای عملیات با مکانیزم Fallback در صورت شکست
        :param function: تابع اصلی که باید اجرا شود
        :param fallback_value: مقدار پیش‌فرض در صورت شکست تابع اصلی
        :param fallback_function: تابع جایگزین که در صورت شکست اجرا می‌شود
        :param args: آرگومان‌های تابع اصلی
        :param kwargs: آرگومان‌های کلیدی تابع اصلی
        :return: نتیجه تابع اصلی در صورت موفقیت، و مقدار `Fallback` در صورت شکست
        """
        try:
            result = function(*args, **kwargs)
            return result
        except Exception as e:
            self.logger.warning(f"⚠️ عملیات {function.__name__} شکست خورد: {e}")

            if fallback_function:
                self.logger.info(f"🔄 اجرای تابع جایگزین {fallback_function.__name__} به عنوان Fallback.")
                return fallback_function(*args, **kwargs)

            if fallback_value is not None:
                self.logger.info(f"✅ مقدار پیش‌فرض {fallback_value} بازگردانده شد.")
                return fallback_value

            self.logger.error("❌ هیچ Fallback مشخص نشده است، مقدار `None` بازگردانده شد.")
            return None
