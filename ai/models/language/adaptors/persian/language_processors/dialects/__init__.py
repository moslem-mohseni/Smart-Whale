# language_processors/dialects/__init__.py

"""
این ماژول نقطه ورود اصلی زیرسیستم لهجه‌بندی است.
تمام کلاس‌ها و ابزارهای مرتبط با پردازش لهجه از جمله:
  - DialectDataAccess: دسترسی متمرکز به داده‌ها (کش، پایگاه داده و غیره)
  - DialectConversionProcessor: تبدیل متن از یک لهجه به لهجه‌ی دیگر
  - DialectLearningProcessor: فرآیند یادگیری از خروجی مدل معلم برای بهبود مدل دانش‌آموز
  - DialectProcessor: هماهنگ‌کننده‌ی کل عملیات تشخیص، تبدیل و یادگیری لهجه

از این فایل برای وارد کردن این کلاس‌ها در سایر بخش‌های پروژه استفاده می‌شود.
"""

from .data_access import DialectDataAccess
from .conversion import DialectConversionProcessor
from .learning import DialectLearningProcessor
from .processor import DialectProcessor

__all__ = [
    "DialectDataAccess",
    "DialectConversionProcessor",
    "DialectLearningProcessor",
    "DialectProcessor",
]
