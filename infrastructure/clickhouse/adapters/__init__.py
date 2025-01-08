# infrastructure/clickhouse/adapters/__init__.py
"""
این ماژول شامل تطبیق‌دهنده‌های ClickHouse است که ارتباط مستقیم با پایگاه داده را مدیریت می‌کنند.
تبدیل عملیات سطح بالا به دستورات ClickHouse در این بخش انجام می‌شود.
"""

from .clickhouse_adapter import ClickHouseAdapter

__all__ = ['ClickHouseAdapter']