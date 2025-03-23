from dataclasses import dataclass


@dataclass
class AdapterConfig:
    """تنظیمات آداپتور ParsBERT"""
    batch_size: int = 16  # اندازه پردازش دسته‌ای
    cache_size: int = 1000  # حداکثر تعداد ورودی‌های کش شده
    max_length: int = 512  # حداکثر طول توکن‌ها برای پردازش
    memory_threshold: float = 0.8  # حد آستانه استفاده از حافظه GPU
    enable_caching: bool = True  # فعال‌سازی کشینگ برای پردازش سریع‌تر

    def validate(self):
        """بررسی مقداردهی تنظیمات"""
        if not (0 < self.batch_size <= 128):
            raise ValueError("batch_size باید بین 1 تا 128 باشد")
        if not (0 < self.cache_size <= 10000):
            raise ValueError("cache_size باید بین 1 تا 10000 باشد")
        if not (64 <= self.max_length <= 1024):
            raise ValueError("max_length باید بین 64 تا 1024 باشد")
        if not (0.1 <= self.memory_threshold <= 1.0):
            raise ValueError("memory_threshold باید بین 0.1 تا 1.0 باشد")
