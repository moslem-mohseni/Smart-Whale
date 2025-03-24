from infrastructure.redis.service.cache_service import CacheService
from typing import Optional, Any

class RedisManager:
    """
    مدیریت کش داده‌ها با استفاده از Redis.
    """
    def __init__(self):
        self.cache_service = CacheService()

    async def connect(self) -> None:
        """ اتصال به Redis """
        await self.cache_service.connect()

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        ذخیره مقدار در کش.

        :param key: کلید داده
        :param value: مقدار داده (JSON / String / Number)
        :param ttl: مدت زمان نگهداری (بر حسب ثانیه)
        """
        await self.cache_service.set(key, value, ttl)

    async def get(self, key: str) -> Optional[Any]:
        """
        دریافت مقدار از کش.

        :param key: کلید داده
        :return: مقدار ذخیره‌شده یا None در صورت عدم وجود
        """
        return await self.cache_service.get(key)

    async def delete(self, key: str) -> bool:
        """
        حذف مقدار از کش.

        :param key: کلید داده
        :return: True در صورت موفقیت، False در غیر این صورت
        """
        return await self.cache_service.delete(key)

    async def set_hash(self, key: str, field: str, value: Any) -> None:
        """
        ذخیره مقدار در HashMap.

        :param key: کلید HashMap
        :param field: فیلد مربوط به داده
        :param value: مقدار داده
        """
        await self.cache_service.hset(key, field, value)

    async def get_hash(self, key: str, field: str) -> Optional[Any]:
        """
        دریافت مقدار از HashMap.

        :param key: کلید HashMap
        :param field: فیلد موردنظر
        :return: مقدار ذخیره‌شده یا None در صورت عدم وجود
        """
        return await self.cache_service.hget(key, field)

    async def expire(self, key: str, ttl: int) -> bool:
        """
        تنظیم زمان انقضا برای یک کلید.

        :param key: کلید داده
        :param ttl: مدت زمان نگهداری (ثانیه)
        :return: True در صورت موفقیت، False در غیر این صورت
        """
        return await self.cache_service.expire(key, ttl)

    async def close(self) -> None:
        """ قطع اتصال از Redis """
        await self.cache_service.disconnect()
