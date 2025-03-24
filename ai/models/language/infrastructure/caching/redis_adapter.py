import logging
import aioredis
from typing import Optional, Any


class RedisAdapter:
    """
    این کلاس ارتباط با Redis را مدیریت کرده و عملیات کش را انجام می‌دهد.
    """

    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.redis = None
        logging.info(f"✅ RedisAdapter مقداردهی شد. [Redis URL: {redis_url}]")

    async def connect(self):
        """
        برقراری اتصال به Redis.
        """
        try:
            self.redis = await aioredis.from_url(self.redis_url, decode_responses=True)
            logging.info("🔌 اتصال به Redis برقرار شد.")
        except Exception as e:
            logging.error(f"❌ خطا در اتصال به Redis: {e}")

    async def disconnect(self):
        """
        قطع اتصال از Redis.
        """
        if self.redis:
            await self.redis.close()
            logging.info("🔌 اتصال Redis بسته شد.")

    async def get(self, key: str) -> Optional[Any]:
        """
        دریافت مقدار ذخیره‌شده از Redis.

        :param key: کلید مورد نظر
        :return: مقدار ذخیره‌شده یا None اگر مقدار وجود نداشته باشد
        """
        try:
            value = await self.redis.get(key)
            if value:
                logging.info(f"📥 مقدار از Redis دریافت شد: {key}")
            else:
                logging.warning(f"⚠️ مقدار `{key}` در Redis وجود ندارد.")
            return value
        except Exception as e:
            logging.error(f"❌ خطا در دریافت مقدار از Redis [{key}]: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: int = 3600):
        """
        ذخیره مقدار در Redis.

        :param key: کلید مورد نظر برای ذخیره
        :param value: مقدار داده‌ای که باید ذخیره شود
        :param ttl: زمان انقضای مقدار در کش (به‌صورت پیش‌فرض ۳۶۰۰ ثانیه)
        """
        try:
            await self.redis.set(key, value, ex=ttl)
            logging.info(f"✅ مقدار در Redis ذخیره شد: {key} (اعتبار: {ttl} ثانیه)")
        except Exception as e:
            logging.error(f"❌ خطا در ذخیره مقدار در Redis [{key}]: {e}")

    async def delete(self, key: str):
        """
        حذف مقدار از Redis.

        :param key: کلید مورد نظر برای حذف
        """
        try:
            await self.redis.delete(key)
            logging.info(f"🗑️ مقدار از Redis حذف شد: {key}")
        except Exception as e:
            logging.error(f"❌ خطا در حذف مقدار از Redis [{key}]: {e}")

    async def flush(self):
        """
        پاک‌سازی کامل کش.
        """
        try:
            await self.redis.flushdb()
            logging.info("🗑️ کل کش Redis پاک شد.")
        except Exception as e:
            logging.error(f"❌ خطا در پاک‌سازی کش Redis: {e}")
