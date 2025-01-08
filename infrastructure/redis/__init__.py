# infrastructure/redis/__init__.py
"""
Redis Cache Service
-----------------
Manages caching layer for high-performance data access.
Includes configuration for Redis clusters and data persistence.
"""

"""
پکیج ردیس برای مدیریت عملیات کش و ذخیره‌سازی موقت داده‌ها.

این پکیج شامل موارد زیر است:
- سرویس مدیریت کش
- مدیریت فضاهای نام
- مدیریت آیتم‌های کش
- ابزارهای نگهداری و مدیریت
"""

from .service import CacheService
from .domain.models import CacheItem, CacheNamespace
from .config.settings import RedisConfig

__version__ = '1.0.0'

__all__ = [
    'CacheService',
    'CacheItem',
    'CacheNamespace',
    'RedisConfig'
]