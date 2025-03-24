import aioredis
from infrastructure.redis.service.cache_service import CacheService

class L3Cache:
    def __init__(self, redis_client: CacheService):
        """
        کش سطح ۳ (L3) - کش دائمی برای داده‌های کم‌مصرف
        :param redis_client: اتصال به سرویس Redis
        """
        self.redis = redis_client

    async def get(self, key: str):
        """ دریافت مقدار از کش L3 """
        return await self.redis.get(key)

    async def set(self, key: str, value: str):
        """ تنظیم مقدار در کش L3 (بدون TTL - داده‌ها دائمی خواهند بود) """
        await self.redis.set(key, value, ttl=None)  # TTL حذف می‌شود و داده دائمی می‌ماند
        await self.redis.redis.execute("PERSIST", key)  # حذف هرگونه TTL موجود

    async def invalidate(self, key: str):
        """ حذف مقدار مشخص از کش L3 """
        await self.redis.delete(key)

    async def clear(self):
        """ پاک‌سازی کامل کش L3 """
        await self.redis.flush()
