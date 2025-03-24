import logging
from typing import Optional, Any
from infrastructure.redis.service.cache_service import CacheService

logger = logging.getLogger(__name__)


class CacheManager:
    """مدیریت کش کوئری‌های TimescaleDB با استفاده از Redis"""

    def __init__(self, cache_service: CacheService):
        """
        مقداردهی اولیه

        Args:
            cache_service (CacheService): سرویس کش از ماژول Redis
        """
        self.cache_service = cache_service

    async def get_cached_result(self, key: str) -> Optional[Any]:
        """
        دریافت نتیجه کش شده برای یک کوئری

        Args:
            key (str): کلید کش

        Returns:
            Optional[Any]: مقدار کش شده یا None در صورت عدم وجود
        """
        cached_data = await self.cache_service.get(key)
        if cached_data:
            logger.info(f"⚡ نتیجه کش برای `{key}` بازیابی شد.")
        return cached_data

    async def cache_result(self, key: str, data: Any, ttl: Optional[int] = None):
        """
        ذخیره نتیجه کوئری در کش

        Args:
            key (str): کلید کش
            data (Any): داده‌ای که باید کش شود
            ttl (Optional[int]): مدت زمان اعتبار کش (پیش‌فرض از تنظیمات ماژول Redis خوانده می‌شود)
        """
        await self.cache_service.set(key, data, ttl=ttl)
        logger.info(f"📥 نتیجه کوئری `{key}` در کش ذخیره شد.")

    async def invalidate_cache(self, key: str):
        """
        حذف یک مقدار از کش

        Args:
            key (str): کلید کش
        """
        await self.cache_service.delete(key)
        logger.info(f"❌ کش `{key}` حذف شد.")

    async def clear_all_cache(self):
        """پاک کردن تمامی کش‌های ذخیره‌شده"""
        await self.cache_service.flush()
        logger.info("🚀 تمامی کش‌های Redis پاک شدند.")
