import logging
import asyncio
from ..adapters.redis_adapter import RedisAdapter

logger = logging.getLogger(__name__)

class CacheCleanup:
    """
    مدیریت پاک‌سازی کلیدهای منقضی‌شده در Redis
    """
    def __init__(self, redis_adapter: RedisAdapter, cleanup_interval: int = 300):
        self.redis_adapter = redis_adapter
        self.cleanup_interval = cleanup_interval  # بازه زمانی برای پاک‌سازی (برحسب ثانیه)
        self._task = None

    async def start(self) -> None:
        """شروع تسک پاک‌سازی به‌صورت دوره‌ای"""
        if not self._task or self._task.done():
            self._task = asyncio.create_task(self._cleanup_loop())
            logger.info("Cache cleanup task started.")

    async def stop(self) -> None:
        """توقف تسک پاک‌سازی"""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            logger.info("Cache cleanup task stopped.")

    async def _cleanup_loop(self) -> None:
        """حلقه پاک‌سازی کلیدهای منقضی‌شده"""
        while True:
            try:
                logger.info("Running cache cleanup...")
                await self.cleanup_expired_keys()
                await asyncio.sleep(self.cleanup_interval)
            except Exception as e:
                logger.error(f"Error in cleanup task: {str(e)}")
                await asyncio.sleep(60)  # در صورت خطا، ۶۰ ثانیه صبر کرده و دوباره اجرا شود

    async def cleanup_expired_keys(self) -> None:
        """پاک‌سازی کلیدهای منقضی‌شده در Redis"""
        try:
            keys = await self.redis_adapter.scan_keys("*")
            for key in keys:
                ttl = await self.redis_adapter.ttl(key)
                if ttl is not None and ttl <= 0:
                    await self.redis_adapter.delete(key)
                    logger.info(f"Deleted expired key: {key}")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

# مثال استفاده:
# cleanup_service = CacheCleanup(redis_adapter)
# await cleanup_service.start()
