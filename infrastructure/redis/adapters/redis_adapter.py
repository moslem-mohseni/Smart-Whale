# infrastructure/redis/adapters/redis_adapter.py

import aioredis
import pickle
import logging
import asyncio
from typing import Any, Optional, Dict, List
from ..config.settings import RedisConfig
from ..domain.models import CacheItem, CacheNamespace
from ...interfaces import CachingInterface, ConnectionError, OperationError

logger = logging.getLogger(__name__)

class RedisAdapter(CachingInterface):
    def __init__(self, config: RedisConfig):
        self.config = config
        self._redis = None
        self._namespaces: Dict[str, CacheNamespace] = {}
        self._max_retries = 3
        self._retry_delay = 1  # seconds

    async def connect(self) -> None:
        """برقراری اتصال به Redis"""
        try:
            if self.config.cluster_mode:
                self._redis = await aioredis.create_redis_cluster(
                    **self.config.get_cluster_params()
                )
            else:
                self._redis = await aioredis.create_redis_pool(
                    **self.config.get_connection_params()
                )
            logger.info("Successfully connected to Redis")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            raise ConnectionError(f"Could not connect to Redis: {str(e)}")

    async def disconnect(self) -> None:
        """قطع اتصال از Redis"""
        if self._redis:
            self._redis.close()
            await self._redis.wait_closed()
            self._redis = None
            logger.info("Disconnected from Redis")

    async def get(self, key: str) -> Optional[Any]:
        """بازیابی مقدار از کش"""
        if not self._redis:
            raise ConnectionError("Not connected to Redis")
        try:
            value = await self._redis.get(key)
            if value is None:
                return None
            return pickle.loads(value)
        except Exception as e:
            logger.error(f"Error getting key {key}: {str(e)}")
            raise OperationError(f"Could not get key {key}: {str(e)}")

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """ذخیره مقدار در کش"""
        if not self._redis:
            raise ConnectionError("Not connected to Redis")
        try:
            serialized = pickle.dumps(value)
            if ttl:
                await self._redis.setex(key, ttl, serialized)
            else:
                await self._redis.set(key, serialized)
        except Exception as e:
            logger.error(f"Error setting key {key}: {str(e)}")
            raise OperationError(f"Could not set key {key}: {str(e)}")

    async def delete(self, key: str) -> bool:
        """حذف کلید از کش"""
        if not self._redis:
            raise ConnectionError("Not connected to Redis")
        try:
            return await self._redis.delete(key) > 0
        except Exception as e:
            logger.error(f"Error deleting key {key}: {str(e)}")
            raise OperationError(f"Could not delete key {key}: {str(e)}")

    async def exists(self, key: str) -> bool:
        """بررسی وجود کلید در کش"""
        if not self._redis:
            raise ConnectionError("Not connected to Redis")
        try:
            return await self._redis.exists(key)
        except Exception as e:
            logger.error(f"Error checking existence of key {key}: {str(e)}")
            raise OperationError(f"Could not check existence of key {key}: {str(e)}")

    async def ttl(self, key: str) -> Optional[int]:
        """دریافت زمان انقضای کلید"""
        if not self._redis:
            raise ConnectionError("Not connected to Redis")
        try:
            ttl = await self._redis.ttl(key)
            return ttl if ttl > -1 else None
        except Exception as e:
            logger.error(f"Error getting TTL for key {key}: {str(e)}")
            raise OperationError(f"Could not get TTL for key {key}: {str(e)}")

    async def scan_keys(self, pattern: str) -> List[str]:
        """جستجوی کلیدها با الگو"""
        if not self._redis:
            raise ConnectionError("Not connected to Redis")
        try:
            keys = []
            cur = b'0'
            while cur:
                cur, keys_batch = await self._redis.scan(
                    cursor=cur,
                    match=pattern,
                    count=100
                )
                keys.extend(keys_batch)
            return [key.decode('utf-8') for key in keys]
        except Exception as e:
            logger.error(f"Error scanning keys with pattern {pattern}: {str(e)}")
            raise OperationError(f"Could not scan keys: {str(e)}")