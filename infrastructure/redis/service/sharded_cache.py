import logging
from typing import List, Any, Optional
from ..adapters.redis_adapter import RedisAdapter
from ..config.settings import RedisConfig

logger = logging.getLogger(__name__)


class ShardedCache:
    """
    مدیریت کش با Sharding برای توزیع داده‌ها در چندین سرور Redis
    """
    def __init__(self, configs: List[RedisConfig]):
        self.shards = [RedisAdapter(config) for config in configs]
        self.shard_count = len(self.shards)

    async def connect(self) -> None:
        """برقراری اتصال با تمام شاردها"""
        for shard in self.shards:
            await shard.connect()
        logger.info("Connected to all Redis shards.")

    async def disconnect(self) -> None:
        """قطع اتصال از تمام شاردها"""
        for shard in self.shards:
            await shard.disconnect()
        logger.info("Disconnected from all Redis shards.")

    def _get_shard(self, key: str) -> RedisAdapter:
        """انتخاب شارد مناسب بر اساس هش کلید"""
        shard_index = hash(key) % self.shard_count
        return self.shards[shard_index]

    async def get(self, key: str) -> Optional[Any]:
        """دریافت مقدار از کش"""
        shard = self._get_shard(key)
        return await shard.get(key)

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """ذخیره مقدار در شارد مناسب"""
        shard = self._get_shard(key)
        await shard.set(key, value, ttl)

    async def delete(self, key: str) -> bool:
        """حذف مقدار از شارد مناسب"""
        shard = self._get_shard(key)
        return await shard.delete(key)

    async def hset(self, key: str, field: str, value: Any) -> None:
        """ذخیره مقدار در HashMap در شارد مناسب"""
        shard = self._get_shard(key)
        await shard.hset(key, field, value)

    async def hget(self, key: str, field: str) -> Optional[Any]:
        """دریافت مقدار از HashMap در شارد مناسب"""
        shard = self._get_shard(key)
        return await shard.hget(key, field)
