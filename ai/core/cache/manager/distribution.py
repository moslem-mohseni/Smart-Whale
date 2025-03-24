import hashlib
from typing import List
from infrastructure.redis.service.sharded_cache import ShardedCache
from infrastructure.redis.service.cache_service import CacheService


class CacheDistribution:
    def __init__(self, redis_shards: List[CacheService]):
        """
        مدیریت توزیع کش بین نودهای مختلف
        :param redis_shards: لیستی از سرویس‌های Redis برای توزیع داده‌ها
        """
        self.sharded_cache = ShardedCache(redis_shards)
        self.is_single_node = len(redis_shards) == 1  # بررسی تعداد نودها

    def _get_shard(self, key: str) -> CacheService:
        """
        انتخاب هوشمند نود Redis برای ذخیره داده
        :param key: کلید کش
        :return: یکی از نودهای Redis برای ذخیره داده
        """
        if self.is_single_node:
            return self.sharded_cache.redis_instances[0]  # تنها نود را برگردان
        else:
            shard_index = int(hashlib.md5(key.encode()).hexdigest(), 16) % len(self.sharded_cache.redis_instances)
            return self.sharded_cache.redis_instances[shard_index]

    async def set(self, key: str, value: str, ttl: int = None):
        """ تنظیم مقدار در نود مشخص‌شده """
        shard = self._get_shard(key)
        await shard.set(key, value, ttl=ttl)

    async def get(self, key: str):
        """ دریافت مقدار از نود مشخص‌شده """
        shard = self._get_shard(key)
        return await shard.get(key)

    async def invalidate(self, key: str):
        """ حذف مقدار مشخص از نود مربوطه """
        shard = self._get_shard(key)
        await shard.delete(key)

    async def rebalance_cache(self):
        """ توزیع مجدد داده‌های کش بین نودها در صورت تغییر ساختار نودها """
        # این متد در آینده برای مهاجرت کش بین نودهای جدید توسعه داده می‌شود
        pass
