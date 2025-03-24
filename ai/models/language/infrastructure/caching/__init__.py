"""
ماژول `caching/` وظیفه‌ی مدیریت کش داده‌های پردازشی مرتبط با زبان را بر عهده دارد.

📌 اجزای اصلی این ماژول:
- `cache_manager.py` → مدیریت عملیات کش و ذخیره‌سازی داده‌های موقت
- `redis_adapter.py` → ارتباط با Redis و اجرای عملیات کش‌گذاری
"""

from .redis_adapter import RedisAdapter
from .cache_manager import CacheManager

# مقداردهی اولیه RedisAdapter
redis_adapter = RedisAdapter(redis_url="redis://localhost:6379")

# مقداردهی اولیه CacheManager
cache_manager = CacheManager(redis_adapter)

__all__ = [
    "redis_adapter",
    "cache_manager",
    "RedisAdapter",
    "CacheManager",
]
