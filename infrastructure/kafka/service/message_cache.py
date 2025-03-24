import hashlib
import logging
import redis.asyncio as redis_asyncio
from ..config.settings import RedisConfig

logger = logging.getLogger(__name__)


async def _generate_hash(message: str) -> str:
    """
    ایجاد هش از پیام برای بررسی یکتایی

    :param message: محتوای پیام
    :return: مقدار هش شده پیام
    """
    return hashlib.sha256(message.encode()).hexdigest()


class MessageCache:
    """
    کش پیام‌های Kafka برای جلوگیری از ارسال پیام‌های تکراری
    """

    def __init__(self, ttl: int = 300):
        """
        مقداردهی اولیه کش پیام‌ها

        :param ttl: مدت زمان نگهداری پیام در کش (ثانیه)
        """
        self.config = RedisConfig()
        self.redis_url = self.config.get_redis_url()
        self.ttl = ttl
        self.redis = None

    async def connect(self):
        """اتصال به Redis"""
        if not self.redis:
            self.redis = redis_asyncio.from_url(self.redis_url, decode_responses=True)
            logger.info(f"Connected to Redis at {self.redis_url}")

    async def close(self):
        """بستن اتصال Redis"""
        if self.redis:
            await self.redis.close()
            logger.info("Redis connection closed.")

    async def is_duplicate(self, message: str) -> bool:
        """
        بررسی می‌کند که آیا پیام قبلاً ارسال شده است یا خیر

        :param message: محتوای پیام
        :return: True اگر پیام تکراری باشد، False در غیر این صورت
        """
        await self.connect()

        message_hash = await _generate_hash(message)

        exists = await self.redis.exists(message_hash)
        if exists:
            logger.info(f"Duplicate message detected: {message}")
            return True

        await self.redis.setex(message_hash, self.ttl, "1")
        return False
