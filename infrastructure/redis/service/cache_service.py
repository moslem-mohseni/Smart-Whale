# infrastructure/redis/service/cache_service.py

from typing import Optional, Any, Dict
from ..config.settings import RedisConfig
from ..domain.models import CacheItem, CacheNamespace
from ..adapters.redis_adapter import RedisAdapter
import logging
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)


class CacheService:
    def __init__(self, config: RedisConfig):
        self.config = config
        self._adapter = RedisAdapter(config)
        self._namespaces: Dict[str, CacheNamespace] = {}
        self._cleanup_task = None
        self._cleanup_interval = 300  # 5 minutes

    def create_namespace(self, namespace: CacheNamespace) -> None:
        """
        ایجاد یک فضای نام جدید

        Args:
            namespace: تنظیمات فضای نام جدید
        """
        self._namespaces[namespace.name] = namespace
        logger.info(f"Created namespace: {namespace.name}")

    def _get_full_key(self, namespace: str, key: str) -> str:
        """
        ساخت کلید کامل با استفاده از namespace
        """
        return f"{namespace}:{key}" if namespace else key

    async def connect(self) -> None:
        """برقراری اتصال به Redis و شروع cleanup task"""
        await self._adapter.connect()
        self._start_cleanup_task()

    def _start_cleanup_task(self) -> None:
        """شروع task دوره‌ای برای cleanup"""
        if not self._cleanup_task or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
            logger.info("Started cleanup task")

    async def _periodic_cleanup(self) -> None:
        """اجرای دوره‌ای عملیات cleanup"""
        while True:
            try:
                await self._cleanup_expired_keys()
                await asyncio.sleep(self._cleanup_interval)
            except Exception as e:
                logger.error(f"Error in cleanup task: {str(e)}")
                await asyncio.sleep(60)  # wait before retry

    async def _cleanup_expired_keys(self) -> None:
        """پاکسازی کلیدهای منقضی شده"""
        if not self._namespaces:  # اگر هیچ namespace نداشته باشیم
            return

        try:
            for namespace in self._namespaces.values():
                pattern = f"{namespace.name}:*"
                keys = await self._adapter.scan_keys(pattern)

                for key in keys:
                    try:
                        ttl = await self._adapter.ttl(key)
                        if ttl is not None and ttl <= 0:
                            success = await self._adapter.delete(key)
                            if success:
                                logger.debug(f"Deleted expired key: {key}")
                            else:
                                logger.warning(f"Failed to delete expired key: {key}")
                    except Exception as e:
                        logger.error(f"Error processing key {key}: {str(e)}")
                        continue
        except Exception as e:
            logger.error(f"Error in cleanup: {str(e)}")

    async def get(self, key: str, namespace: Optional[str] = None) -> Optional[Any]:
        """
        بازیابی یک مقدار از کش
        """
        full_key = self._get_full_key(namespace, key) if namespace else key
        try:
            value = await self._adapter.get(full_key)
            if isinstance(value, CacheItem):
                if value.ttl and (datetime.now() - value.created_at).total_seconds() > value.ttl:
                    await self._adapter.delete(full_key)
                    return None
                return value.value
            return value
        except Exception as e:
            logger.error(f"Error retrieving key {full_key}: {str(e)}")
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None, namespace: Optional[str] = None) -> None:
        """
        ذخیره یک مقدار در کش
        """
        full_key = self._get_full_key(namespace, key) if namespace else key

        if namespace and namespace in self._namespaces:
            namespace_config = self._namespaces[namespace]
            if ttl is None:
                ttl = namespace_config.default_ttl

        cache_item = CacheItem(
            key=full_key,
            value=value,
            ttl=ttl,
            created_at=datetime.now()
        )

        await self._adapter.set(full_key, cache_item, ttl)

    async def disconnect(self) -> None:
        """قطع اتصال از Redis و توقف cleanup task"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
            logger.info("Stopped cleanup task")

        await self._adapter.disconnect()