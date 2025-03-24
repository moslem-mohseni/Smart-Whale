from infrastructure.redis.service.cache_service import CacheService
from infrastructure.redis.config.settings import RedisConfig


class CacheManager:
    """
    مدیریت کش متادیتای فایل‌ها و هش‌ها در Redis
    """
    def __init__(self):
        config = RedisConfig()  # دریافت تنظیمات از ماژول Redis
        self.cache_service = CacheService(config)

    async def set(self, key: str, value: str, ttl: int = 3600):
        """ذخیره مقدار در کش با زمان انقضا"""
        await self.cache_service.set(key, value, ttl)

    async def get(self, key: str):
        """دریافت مقدار از کش"""
        return await self.cache_service.get(key)

    async def delete(self, key: str):
        """حذف کلید از کش"""
        await self.cache_service.delete(key)
