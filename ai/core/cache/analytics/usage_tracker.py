import time
from infrastructure.redis.service.cache_service import CacheService

class CacheUsageTracker:
    def __init__(self, redis_client: CacheService, tracking_key="cache_usage_tracker"):
        """
        ردیابی استفاده از کش
        :param redis_client: سرویس Redis برای مدیریت کش
        :param tracking_key: کلید ذخیره اطلاعات استفاده از کش
        """
        self.redis = redis_client
        self.tracking_key = tracking_key  # نام HashMap در Redis برای ذخیره داده‌ها

    async def track_usage(self, key: str):
        """
        ثبت تعداد استفاده از یک کلید کش
        :param key: کلید کش که استفاده شده است
        """
        timestamp = int(time.time())  # ثبت زمان آخرین استفاده
        await self.redis.redis.hincrby(self.tracking_key, key, 1)
        await self.redis.redis.hset(f"{self.tracking_key}_timestamps", key, timestamp)

    async def get_most_used_keys(self, top_n: int = 5):
        """
        دریافت لیست پرکاربردترین کلیدهای کش
        :param top_n: تعداد کلیدهای برتر برای نمایش
        :return: لیست کلیدهای کش به ترتیب بیشترین استفاده
        """
        usage_data = await self.redis.redis.hgetall(self.tracking_key)
        sorted_usage = sorted(usage_data.items(), key=lambda x: int(x[1]), reverse=True)
        return sorted_usage[:top_n]

    async def cleanup_unused_keys(self, expiration_time: int = 86400):
        """
        حذف کلیدهای کم‌استفاده که برای مدت طولانی استفاده نشده‌اند
        :param expiration_time: مدت زمان (ثانیه) که اگر از کلید استفاده نشده باشد، حذف شود
        """
        current_time = int(time.time())
        timestamps = await self.redis.redis.hgetall(f"{self.tracking_key}_timestamps")

        for key, last_used in timestamps.items():
            if current_time - int(last_used) > expiration_time:
                await self.redis.redis.hdel(self.tracking_key, key)
                await self.redis.redis.hdel(f"{self.tracking_key}_timestamps", key)
                await self.redis.delete(key)  # حذف کلید از کش
