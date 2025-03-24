from cachetools import LRUCache, TTLCache
from typing import Optional, Any

class MemoryCache:
    """
    مدیریت کش داده‌های پرکاربرد در حافظه (Memory Cache).
    این ماژول از دو نوع کش استفاده می‌کند:
    - LRUCache: برای داده‌های کم‌حجم با کنترل ظرفیت
    - TTLCache: برای داده‌هایی که نیاز به حذف خودکار دارند
    """

    def __init__(self, max_size: int = 1000, ttl: int = 300):
        """
        مقداردهی اولیه کش.

        :param max_size: حداکثر تعداد آیتم‌های ذخیره‌شده در LRUCache
        :param ttl: مدت زمان نگهداری آیتم‌ها در TTLCache (بر حسب ثانیه)
        """
        self.lru_cache = LRUCache(maxsize=max_size)
        self.ttl_cache = TTLCache(maxsize=max_size, ttl=ttl)

    def set(self, key: str, value: Any, use_ttl: bool = False) -> None:
        """
        ذخیره مقدار در کش.

        :param key: کلید داده
        :param value: مقدار داده
        :param use_ttl: اگر True باشد، مقدار در TTLCache ذخیره می‌شود
        """
        if use_ttl:
            self.ttl_cache[key] = value
        else:
            self.lru_cache[key] = value

    def get(self, key: str) -> Optional[Any]:
        """
        دریافت مقدار از کش.

        :param key: کلید داده
        :return: مقدار ذخیره‌شده یا None در صورت عدم وجود
        """
        return self.lru_cache.get(key) or self.ttl_cache.get(key)

    def delete(self, key: str) -> bool:
        """
        حذف مقدار از کش.

        :param key: کلید داده
        :return: True در صورت حذف موفق، False در غیر این صورت
        """
        if key in self.lru_cache:
            del self.lru_cache[key]
            return True
        if key in self.ttl_cache:
            del self.ttl_cache[key]
            return True
        return False

    def clear(self) -> None:
        """
        پاک‌سازی کامل کش.
        """
        self.lru_cache.clear()
        self.ttl_cache.clear()
