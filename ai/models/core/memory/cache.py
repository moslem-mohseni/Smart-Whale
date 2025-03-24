import time
import functools
from typing import Any, Dict, Optional
from collections import OrderedDict

class Cache:
    """
    پیاده‌سازی کش چندسطحی برای ذخیره داده‌های پردازشی و جلوگیری از پردازش‌های تکراری.
    """

    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        """
        مقداردهی اولیه کش.
        :param max_size: حداکثر تعداد آیتم‌هایی که کش می‌تواند ذخیره کند.
        :param ttl: زمان نگهداری داده در کش بر حسب ثانیه.
        """
        self.max_size = max_size  # حداکثر تعداد آیتم‌های کش
        self.ttl = ttl  # زمان زنده بودن داده‌های کش
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        """
        دریافت مقدار ذخیره‌شده در کش.
        :param key: کلید مربوط به مقدار ذخیره‌شده.
        :return: مقدار ذخیره‌شده در صورت معتبر بودن، در غیر اینصورت None.
        """
        if key in self.cache:
            data = self.cache.pop(key)  # حذف و دریافت داده برای به‌روزرسانی موقعیت آن در LRU
            if time.time() - data["timestamp"] < self.ttl:
                self.cache[key] = data  # برگرداندن داده به کش برای به‌روزرسانی ترتیب
                return data["value"]
            else:
                del self.cache[key]  # حذف داده منقضی‌شده
        return None

    def set(self, key: str, value: Any):
        """
        ذخیره مقدار در کش.
        :param key: کلید مقدار ذخیره‌شده.
        :param value: مقدار داده‌ای که باید ذخیره شود.
        """
        if key in self.cache:
            self.cache.pop(key)
        elif len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)  # حذف قدیمی‌ترین مقدار (FIFO در صورت پر شدن)

        self.cache[key] = {"value": value, "timestamp": time.time()}

    def clear(self):
        """
        پاک‌سازی کامل کش.
        """
        self.cache.clear()

    def cache_decorator(self, ttl: Optional[int] = None):
        """
        دکوراتور برای کش‌کردن خروجی توابع پرهزینه پردازشی.
        :param ttl: مدت زمان معتبر بودن داده کش‌شده (اختیاری).
        :return: خروجی کش‌شده تابع.
        """
        def decorator(func):
            cache_instance = self
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                key = f"{func.__name__}_{args}_{kwargs}"
                cached_value = cache_instance.get(key)
                if cached_value is not None:
                    return cached_value
                result = func(*args, **kwargs)
                cache_instance.set(key, result)
                return result
            return wrapper
        return decorator
