from ai.core.cache.hierarchical.l1_cache import L1Cache
from ai.core.cache.hierarchical.l2_cache import L2Cache
from ai.core.cache.hierarchical.l3_cache import L3Cache
from infrastructure.redis.service.cache_service import CacheService


class CacheInvalidation:
    def __init__(self, redis_client: CacheService):
        """
        مدیریت مکانیزم‌های حذف کش در تمامی سطوح
        :param redis_client: سرویس Redis برای مدیریت کش
        """
        self.l1_cache = L1Cache(redis_client)
        self.l2_cache = L2Cache(redis_client)
        self.l3_cache = L3Cache(redis_client)
        self.redis = redis_client

    async def invalidate_key(self, key: str):
        """ حذف مقدار مشخص از تمام سطوح کش """
        await self.l1_cache.invalidate(key)
        await self.l2_cache.invalidate(key)
        await self.l3_cache.invalidate(key)

    async def pattern_invalidate(self, pattern: str):
        """ حذف تمام مقادیر مطابق یک الگو از کش """
        keys = await self.redis.redis.keys(pattern)  # دریافت کلیدهای مطابق الگو
        for key in keys:
            await self.invalidate_key(key)

    async def cleanup_expired(self):
        """ پاک‌سازی کلیدهای منقضی شده از کش """
        # Redis به‌طور خودکار کلیدهای دارای TTL منقضی‌شده را پاک می‌کند
        # این متد برای اعتبارسنجی مجدد در صورت نیاز طراحی شده است
        pass  # در صورت نیاز می‌توان این متد را با بررسی TTL پیاده‌سازی کرد

    async def flush_all(self):
        """ پاک‌سازی کلی کش (تمام سطوح L1، L2 و L3) """
        await self.l1_cache.clear()
        await self.l2_cache.clear()
        await self.l3_cache.clear()
        await self.redis.flush()  # حذف همه کلیدها از Redis
