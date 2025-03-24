import json
from typing import Any, Optional
from infrastructure.redis.service.cache_service import CacheService
from infrastructure.redis.config.settings import RedisConfig


class CacheManager:
    """مدیریت کش برای افزایش سرعت جستجوهای برداری"""

    def __init__(self):
        redis_config = RedisConfig()  # مقداردهی تنظیمات Redis
        self.cache_service = CacheService(redis_config)  # ارسال تنظیمات به CacheService

    async def get_cached_result(self, key: str) -> Optional[Any]:
        """دریافت نتیجه کش‌شده از Redis"""
        cached_data = await self.cache_service.get(key)
        if cached_data:
            print(f"✅ کش یافت شد: {key}")
            return json.loads(cached_data)
        print(f"❌ کشی برای {key} یافت نشد.")
        return None

    async def cache_result(self, key: str, value: Any, ttl: Optional[int] = 3600):
        """ذخیره نتیجه در Redis"""
        await self.cache_service.set(key, json.dumps(value), ttl=ttl)
        print(f"💾 نتیجه جستجو برای {key} در کش ذخیره شد.")

    async def delete_cache(self, key: str):
        """حذف یک مقدار از کش"""
        await self.cache_service.delete(key)
        print(f"🗑️ مقدار {key} از کش حذف شد.")

    async def flush_cache(self):
        """پاکسازی تمام کش‌ها"""
        await self.cache_service.flush()
        print("🚀 تمام کش‌های Redis پاکسازی شدند.")
