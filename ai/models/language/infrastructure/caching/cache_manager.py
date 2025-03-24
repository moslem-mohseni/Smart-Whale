import logging
from typing import Optional, Any
from ai.models.language.infrastructure.caching.redis_adapter import RedisAdapter


class CacheManager:
    """
    این کلاس مدیریت کش داده‌های پردازشی مرتبط با زبان را بر عهده دارد.
    """

    def __init__(self, redis_adapter: RedisAdapter):
        self.redis = redis_adapter
        logging.info("✅ CacheManager مقداردهی شد.")

    async def get_cached_result(self, key: str) -> Optional[Any]:
        """
        بازیابی نتیجه‌ی کش‌شده از Redis.

        :param key: کلید ذخیره‌شده در کش
        :return: مقدار ذخیره‌شده یا None اگر مقدار وجود نداشته باشد
        """
        cached_value = await self.redis.get(key)
        if cached_value:
            logging.info(f"📥 مقدار کش بازیابی شد: {key}")
        else:
            logging.warning(f"⚠️ مقدار مورد نظر در کش وجود ندارد: {key}")
        return cached_value

    async def cache_result(self, key: str, value: Any, ttl: int = 3600):
        """
        ذخیره‌ی نتیجه در کش.

        :param key: کلید مورد نظر برای ذخیره
        :param value: مقدار داده‌ای که باید ذخیره شود
        :param ttl: زمان اعتبار کش (به‌صورت پیش‌فرض یک ساعت)
        """
        await self.redis.set(key, value, ttl)
        logging.info(f"✅ مقدار در کش ذخیره شد: {key} (اعتبار: {ttl} ثانیه)")

    async def delete_cached_result(self, key: str):
        """
        حذف مقدار ذخیره‌شده از کش.

        :param key: کلید مورد نظر برای حذف
        """
        await self.redis.delete(key)
        logging.info(f"🗑️ مقدار از کش حذف شد: {key}")

    async def flush_cache(self):
        """
        پاک‌سازی کامل کش.
        """
        await self.redis.flush()
        logging.info("🗑️ کل کش سیستم پاک شد.")
