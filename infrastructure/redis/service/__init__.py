# infrastructure/redis/service/__init__.py
"""
این ماژول سرویس‌های سطح بالا برای کار با ردیس را فراهم می‌کند.
CacheService اصلی که رابط ساده‌تری برای کار با کش ارائه می‌دهد در این بخش تعریف شده است.
"""

from .cache_service import CacheService

__all__ = ['CacheService']