# infrastructure/clickhouse/config/__init__.py
"""
این ماژول شامل کلاس‌ها و توابع مربوط به پیکربندی ClickHouse است.
تنظیمات اتصال، پارامترهای عملکردی و سایر تنظیمات در این بخش مدیریت می‌شوند.
"""

from .config import ClickHouseConfig

__all__ = ['ClickHouseConfig']
