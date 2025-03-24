from infrastructure.redis.service.cache_service import CacheService
from infrastructure.redis.config.settings import RedisConfig


class HashCache:
    """
    مدیریت کش هش فایل‌ها در Redis
    """
    def __init__(self):
        config = RedisConfig()  # دریافت تنظیمات از ماژول Redis
        self.cache_service = CacheService(config)

    async def store_file_hash(self, file_name: str, file_hash: str, ttl: int = 86400):
        """ذخیره هش فایل در کش با زمان انقضا"""
        await self.cache_service.set(f"file_hash:{file_name}", file_hash, ttl)

    async def get_file_hash(self, file_name: str):
        """بازیابی هش فایل از کش"""
        return await self.cache_service.get(f"file_hash:{file_name}")

    async def delete_file_hash(self, file_name: str):
        """حذف هش فایل از کش"""
        await self.cache_service.delete(f"file_hash:{file_name}")
