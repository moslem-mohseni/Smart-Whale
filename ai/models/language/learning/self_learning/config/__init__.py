"""
config Package
-------------------
این پکیج شامل تنظیمات پیش‌فرض و پارامترهای مقیاس‌بندی سیستم خودآموزی است:
  - default_config: تنظیمات پیش‌فرض ماژول Self-Learning.
  - scaling_parameters: پارامترهای مقیاس‌بندی و بهینه‌سازی سیستم.

تمامی تنظیمات به صورت نهایی و عملیاتی تعریف شده‌اند.
"""

from .default_config import DEFAULT_CONFIG
from .scaling_parameters import SCALING_PARAMETERS

__all__ = [
    "DEFAULT_CONFIG",
    "SCALING_PARAMETERS"
]
