# infrastructure/clickhouse/migrations/__init__.py
"""
این ماژول شامل اسکریپت‌های مربوط به تغییرات ساختاری در پایگاه داده است.
هر تغییر در ساختار داده‌ها باید از طریق این اسکریپت‌ها مدیریت شود تا قابل پیگیری باشد.
"""

from .v001_initial_schema import upgrade, downgrade

__all__ = ['upgrade', 'downgrade']