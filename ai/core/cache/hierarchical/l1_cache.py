import aioredis
from infrastructure.redis.service.cache_service import CacheService

class L1Cache:
    def __init__(self, redis_client: CacheService, ttl: int = 10):
        """
        کش سطح ۱ (L1) - سریع‌ترین سطح کش
        :param redis_client: اتصال به سرویس Redis
        :param ttl: مدت زمان نگهداری داده در کش (ثانیه)
        """
        self.redis = redis_client
        self.ttl = ttl

    async def get(self, key: str):
        """ دریافت مقدار از کش L1 """
        return await self.redis.get(key)

    async def set(self, key: str, value: str):
        """ تنظیم مقدار در کش L1 """
        await self.redis.set(key, value, ttl=self.ttl)

    async def invalidate(self, key: str):
        """ حذف مقدار مشخص از کش L1 """
        await self.redis.delete(key)

    async def clear(self):
        """ پاک‌سازی کامل کش L1 """
        await self.redis.flush()
