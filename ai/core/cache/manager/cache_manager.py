from ai.core.cache.hierarchical.l2_cache import L2Cache
from ai.core.cache.hierarchical.l3_cache import L3Cache
from infrastructure.redis.service.cache_service import CacheService


class CacheManager:
    def __init__(self, redis_client=None):
        if redis_client is None:
            from infrastructure.redis.service.cache_service import CacheService
            redis_client = CacheService
        """
        مدیریت کلی کش در سه سطح L1، L2 و L3
        :param redis_client: سرویس Redis برای مدیریت کش
        """
        self.l1_cache = L1Cache(redis_client)
        self.l2_cache = L2Cache(redis_client)
        self.l3_cache = L3Cache(redis_client)

    async def get(self, key: str):
        """ دریافت مقدار از کش (اول L1، بعد L2 و در نهایت L3) """
        value = await self.l1_cache.get(key)
        if value is not None:
            return value

        value = await self.l2_cache.get(key)
        if value is not None:
            await self.l1_cache.set(key, value)  # ذخیره در L1 برای دسترسی سریع‌تر
            return value

        value = await self.l3_cache.get(key)
        if value is not None:
            await self.l2_cache.set(key, value)  # ذخیره در L2
            await self.l1_cache.set(key, value)  # ذخیره در L1
            return value

        return None  # مقدار یافت نشد

    async def set(self, key: str, value: str):
        """ تنظیم مقدار در تمام سطوح کش """
        await self.l1_cache.set(key, value)
        await self.l2_cache.set(key, value)
        await self.l3_cache.set(key, value)

    async def invalidate(self, key: str):
        """ حذف مقدار از تمام سطوح کش """
        await self.l1_cache.invalidate(key)
        await self.l2_cache.invalidate(key)
        await self.l3_cache.invalidate(key)

    async def clear_all(self):
        """ پاک‌سازی کامل تمام کش‌ها """
        await self.l1_cache.clear()
        await self.l2_cache.clear()
        await self.l3_cache.clear()
