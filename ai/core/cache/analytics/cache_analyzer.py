from infrastructure.redis.service.cache_service import CacheService


class CacheAnalyzer:
    def __init__(self, redis_client: CacheService):
        """
        تحلیل‌گر عملکرد کش
        :param redis_client: سرویس Redis برای مدیریت کش
        """
        self.redis = redis_client

    async def get_cache_hit_ratio(self):
        """
        محاسبه نرخ برخورد کش (Cache Hit Ratio)
        :return: نسبت hit/miss به‌صورت درصدی
        """
        stats = await self.redis.redis.info("stats")
        keyspace_hits = int(stats.get("keyspace_hits", 0))
        keyspace_misses = int(stats.get("keyspace_misses", 0))
        total_requests = keyspace_hits + keyspace_misses

        if total_requests == 0:
            return 0.0  # هیچ داده‌ای در کش استفاده نشده است

        return round((keyspace_hits / total_requests) * 100, 2)

    async def get_memory_usage(self):
        """
        دریافت مقدار حافظه مصرف‌شده در Redis برای کش
        :return: میزان مصرف حافظه (بایت)
        """
        stats = await self.redis.redis.info("memory")
        return int(stats.get("used_memory", 0))

    async def suggest_cache_optimization(self):
        """
        ارائه پیشنهادات بهینه‌سازی بر اساس عملکرد کش
        :return: دیکشنری شامل پیشنهادات بهینه‌سازی
        """
        hit_ratio = await self.get_cache_hit_ratio()
        memory_usage = await self.get_memory_usage()

        suggestions = {}

        if hit_ratio < 50:
            suggestions["increase_cache_size"] = "نرخ برخورد کش پایین است. پیشنهاد می‌شود اندازه کش افزایش یابد."

        if memory_usage > 500 * 1024 * 1024:  # بیش از 500MB مصرف کش
            suggestions["reduce_ttl"] = "حافظه کش بیش از حد مصرف شده است. پیشنهاد می‌شود TTL کاهش یابد یا سیاست حذف داده بهینه شود."

        return suggestions
