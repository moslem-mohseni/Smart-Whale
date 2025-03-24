import logging
from typing import Any, Optional
from ..adapters.redis_adapter import RedisAdapter
from ..config.settings import RedisConfig

logger = logging.getLogger(__name__)


class FallbackCache:
    """
    مدیریت مکانیزم پشتیبان برای Redis در صورت خرابی سرور اصلی
    """
    def __init__(self, primary_config: RedisConfig, secondary_config: RedisConfig):
        self.primary = RedisAdapter(primary_config)
        self.secondary = RedisAdapter(secondary_config)

    async def connect(self) -> None:
        """برقراری اتصال با هر دو سرور"""
        await self.primary.connect()
        await self.secondary.connect()
        logger.info("Connected to primary and secondary Redis instances.")

    async def disconnect(self) -> None:
        """قطع اتصال از هر دو سرور"""
        await self.primary.disconnect()
        await self.secondary.disconnect()
        logger.info("Disconnected from both Redis instances.")

    async def get(self, key: str) -> Optional[Any]:
        """دریافت مقدار از کش، ابتدا از primary و در صورت خطا از secondary"""
        try:
            value = await self.primary.get(key)
            if value is not None:
                return value
        except Exception as e:
            logger.warning(f"Primary Redis failed: {str(e)}. Falling back to secondary.")

        return await self.secondary.get(key)

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """ذخیره مقدار در هر دو سرور"""
        try:
            await self.primary.set(key, value, ttl)
        except Exception as e:
            logger.warning(f"Primary Redis failed to set key {key}: {str(e)}")

        try:
            await self.secondary.set(key, value, ttl)
        except Exception as e:
            logger.warning(f"Secondary Redis failed to set key {key}: {str(e)}")

    async def delete(self, key: str) -> bool:
        """حذف مقدار از هر دو سرور"""
        primary_deleted = False
        try:
            primary_deleted = await self.primary.delete(key)
        except Exception as e:
            logger.warning(f"Primary Redis failed to delete key {key}: {str(e)}")

        secondary_deleted = False
        try:
            secondary_deleted = await self.secondary.delete(key)
        except Exception as e:
            logger.warning(f"Secondary Redis failed to delete key {key}: {str(e)}")

        return primary_deleted or secondary_deleted
