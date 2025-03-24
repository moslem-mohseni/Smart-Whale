import aioredis
from infrastructure.redis.service.cache_service import CacheService

class L2Cache:
    def __init__(self, redis_client: CacheService, ttl: int = 300):
        """
        کش سطح ۲ (L2) - کش میانی برای داده‌های عمومی
        :param redis_client: اتصال به سرویس Redis
        :param ttl: مدت زمان نگهداری داده در کش (ثانیه)
        """
        self.redis = redis_client
        self.ttl = ttl

    async def get(self, key: str):
        """ دریافت مقدار از کش L2 """
        return await self.redis.get(key)

    async def set(self, key: str, value: str):
        """ تنظیم مقدار در کش L2 """
        await self.redis.set(key, value, ttl=self.ttl)

    async def invalidate(self, key: str):
        """ حذف مقدار مشخص از کش L2 """
        await self.redis.delete(key)

    async def clear(self):
        """ پاک‌سازی کامل کش L2 """
        await self.redis.flush()
