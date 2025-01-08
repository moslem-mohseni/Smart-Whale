# infrastructure/redis/adapters/__init__.py
"""
این ماژول شامل تطبیق‌دهنده‌های ردیس است که مسئول ارتباط مستقیم با سرور ردیس هستند.
RedisAdapter اصلی که پیاده‌سازی CachingInterface است در این بخش تعریف شده است.
"""

from .redis_adapter import RedisAdapter

__all__ = ['RedisAdapter']
