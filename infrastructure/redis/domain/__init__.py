# infrastructure/redis/domain/__init__.py
"""
این ماژول شامل مدل‌های دامنه برای کار با ردیس است.
ساختارهای داده پایه مانند CacheItem و CacheNamespace در اینجا تعریف می‌شوند.
"""

from .models import CacheItem, CacheNamespace

__all__ = ['CacheItem', 'CacheNamespace']