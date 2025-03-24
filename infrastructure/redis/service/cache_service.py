import logging
import asyncio
from typing import Optional, Any
from ..adapters.redis_adapter import RedisAdapter
from ..config.settings import RedisConfig

logger = logging.getLogger(__name__)


class CacheService:
    """
    سرویس مدیریت کش برای ارتباط آسان‌تر با Redis
    """

    def __init__(self,  config: Optional[RedisConfig] = None):
        self._adapter = RedisAdapter(config=config or RedisConfig())
        self._cleanup_task = None
        self._cleanup_interval = 300  # هر 5 دقیقه پاک‌سازی اجرا می‌شود
        self.default_ttl = int(getattr(config, "default_ttl", 300))  # مقدار پیش‌فرض TTL

    async def connect(self) -> None:
        """برقراری اتصال به Redis"""
        await self._adapter.connect()
        self._start_cleanup_task()

    async def disconnect(self) -> None:
        """قطع اتصال از Redis و توقف تسک‌های پس‌زمینه"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        await self._adapter.disconnect()

    async def get(self, key: str) -> Optional[Any]:
        """دریافت مقدار از کش"""
        return await self._adapter.get(key)

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """ذخیره مقدار در کش با امکان تنظیم زمان انقضا"""
        ttl = ttl or self.default_ttl  # استفاده از TTL پیش‌فرض در صورت عدم تعیین TTL
        await self._adapter.set(key, value, ttl)

    async def delete(self, key: str) -> bool:
        """حذف مقدار از کش"""
        return await self._adapter.delete(key)

    async def flush(self) -> None:
        """پاک‌سازی کامل کش"""
        await self._adapter.flush()
        logger.info("All cache keys have been flushed.")

    async def hset(self, key: str, field: str, value: Any) -> None:
        """ذخیره مقدار در HashMap"""
        await self._adapter.hset(key, field, value)

    async def hget(self, key: str, field: str) -> Optional[Any]:
        """دریافت مقدار از HashMap"""
        return await self._adapter.hget(key, field)

    async def _periodic_cleanup(self) -> None:
        """پاکسازی دوره‌ای کلیدهای منقضی‌شده"""
        while True:
            try:
                logger.info("Executing periodic cache cleanup...")
                await asyncio.sleep(self._cleanup_interval)
            except Exception as e:
                logger.error(f"Error in cleanup task: {str(e)}")
                await asyncio.sleep(60)  # در صورت خطا، 60 ثانیه صبر کرده و دوباره اجرا شود

    def _start_cleanup_task(self) -> None:
        """شروع تسک دوره‌ای پاک‌سازی"""
        if not self._cleanup_task or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
            logger.info("Started periodic cache cleanup task.")
