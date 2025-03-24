import aioredis
from typing import Optional, Any, List

class DistributedCache:
    """
    مدیریت کش توزیع‌شده با استفاده از Redis Cluster.
    """

    def __init__(self, redis_nodes: List[str]):
        """
        مقداردهی اولیه کش.

        :param redis_nodes: لیست آدرس‌های نودهای Redis Cluster
        """
        self.redis_nodes = redis_nodes
        self.redis_cluster = None

    async def connect(self) -> None:
        """ اتصال به Redis Cluster """
        self.redis_cluster = await aioredis.from_cluster_nodes(
            [(node.split(":")[0], int(node.split(":")[1])) for node in self.redis_nodes]
        )

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        ذخیره مقدار در کش توزیع‌شده.

        :param key: کلید داده
        :param value: مقدار داده
        :param ttl: مدت زمان نگهداری (ثانیه)
        """
        if ttl:
            await self.redis_cluster.set(key, value, expire=ttl)
        else:
            await self.redis_cluster.set(key, value)

    async def get(self, key: str) -> Optional[Any]:
        """
        دریافت مقدار از کش توزیع‌شده.

        :param key: کلید داده
        :return: مقدار ذخیره‌شده یا None در صورت عدم وجود
        """
        return await self.redis_cluster.get(key)

    async def delete(self, key: str) -> bool:
        """
        حذف مقدار از کش.

        :param key: کلید داده
        :return: True در صورت موفقیت، False در غیر این صورت
        """
        return await self.redis_cluster.delete(key) > 0

    async def keys(self, pattern: str = "*") -> List[str]:
        """
        دریافت کلیدهای موجود در کش.

        :param pattern: الگوی جستجو برای کلیدها (پیش‌فرض: همه کلیدها)
        :return: لیست کلیدهای موجود
        """
        return await self.redis_cluster.keys(pattern)

    async def expire(self, key: str, ttl: int) -> bool:
        """
        تنظیم زمان انقضا برای یک کلید.

        :param key: کلید داده
        :param ttl: مدت زمان نگهداری (ثانیه)
        :return: True در صورت موفقیت، False در غیر این صورت
        """
        return await self.redis_cluster.expire(key, ttl)

    async def close(self) -> None:
        """ قطع اتصال از Redis Cluster """
        await self.redis_cluster.close()
