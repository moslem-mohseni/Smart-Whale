# persian/language_processors/utils/text_normalization.py

"""
ماژول text_normalization.py

این ماژول شامل کلاس TextNormalizer است که وظیفه نرمال‌سازی و بهینه‌سازی متن‌های فارسی را بر عهده دارد.
ابتدا تلاش می‌کند از کتابخانه hazm استفاده کند؛ در صورت عدم دسترسی، از پیاده‌سازی داخلی پیشرفته بهره می‌برد.
"""

import re
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# تلاش برای وارد کردن hazm
try:
    from hazm import Normalizer as HazmNormalizer

    USE_HAZM = True
    logger.info("کتابخانه hazm با موفقیت بارگذاری شد.")
except ImportError:
    USE_HAZM = False
    HazmNormalizer = None  # تعریف پیش‌فرض در صورت عدم وجود hazm
    logger.warning("کتابخانه hazm یافت نشد؛ از پیاده‌سازی پیشرفته‌ی داخلی استفاده خواهد شد.")


class TextNormalizer:
    def __init__(self):
        """
        مقداردهی اولیه TextNormalizer.
        اگر hazm موجود باشد، از آن استفاده می‌شود؛ در غیر این صورت از پیاده‌سازی داخلی پیشرفته بهره می‌بریم.
        """
        if USE_HAZM and HazmNormalizer:
            self.hazm_normalizer = HazmNormalizer()
        else:
            self.hazm_normalizer = None
        # تعریف الگوهای جایگزینی برای حالت fallback
        self.replacements = {
            'ي': 'ی',
            'ك': 'ک',
            'ۀ': 'ه',
            '٤': '۴', '٥': '۵', '٦': '۶',
            # تبدیل اعداد انگلیسی به فارسی
            '0': '۰', '1': '۱', '2': '۲', '3': '۳', '4': '۴',
            '5': '۵', '6': '۶', '7': '۷', '8': '۸', '9': '۹'
        }
        # الگوی حذف فاصله‌های اضافی
        self.whitespace_pattern = re.compile(r'\s+')

    def normalize(self, text: str) -> str:
        """
        نرمال‌سازی متن فارسی.
        ابتدا اگر hazm موجود باشد، از آن بهره می‌بریم و سپس بهبودهای تکمیلی انجام می‌شود.
        در غیر این صورت، از پیاده‌سازی داخلی استفاده می‌شود.

        Args:
            text (str): متن ورودی

        Returns:
            str: متن نرمال‌شده
        """
        if USE_HAZM and self.hazm_normalizer:
            try:
                # استفاده از hazm برای نرمال‌سازی اولیه
                normalized = self.hazm_normalizer.normalize(text)
            except Exception as e:
                logger.error(f"خطا در نرمال‌سازی با hazm: {e}")
                normalized = self._fallback_normalize(text)
        else:
            normalized = self._fallback_normalize(text)

        # بهبودهای تکمیلی: حذف فاصله‌های اضافی و اصلاح علائم نگارشی
        normalized = self.whitespace_pattern.sub(' ', normalized)
        normalized = normalized.strip()
        return normalized

    def _fallback_normalize(self, text: str) -> str:
        """
        پیاده‌سازی داخلی پیشرفته نرمال‌سازی در صورت عدم دسترسی به hazm.
        این پیاده‌سازی شامل جایگزینی کاراکترهای عربی به فارسی، حذف فاصله‌های اضافی و اصلاح نگارش است.

        Args:
            text (str): متن ورودی

        Returns:
            str: متن نرمال‌شده
        """
        normalized = text
        # جایگزینی کاراکترهای رایج
        for old, new in self.replacements.items():
            normalized = normalized.replace(old, new)

        # حذف علائم ناخواسته و اصلاح فاصله‌گذاری
        # حفظ علائم نگارشی مهم و اصلاح فاصله بعد از آنها
        normalized = re.sub(r'([.,!?؛:؟])\s*', r'\1 ', normalized)
        normalized = self.whitespace_pattern.sub(' ', normalized)
        return normalized.strip()


if __name__ == "__main__":
    sample_text = "سلام! امروز هوا خیلی عالی است... آیا شما هم موافقید؟ 12345"
    normalizer = TextNormalizer()
    result = normalizer.normalize(sample_text)
    print("متن نرمال‌شده:", result)
