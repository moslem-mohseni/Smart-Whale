import asyncio
from typing import Optional
from ..adapters.redis_adapter import RedisAdapter


class RateLimiter:
    """
    مدیریت نرخ درخواست‌ها به Redis برای جلوگیری از سوءاستفاده و حملات DoS
    """

    def __init__(self, redis_adapter: RedisAdapter, max_requests: int, window_seconds: int):
        self.redis_adapter = redis_adapter
        self.max_requests = max_requests
        self.window_seconds = window_seconds

    async def is_allowed(self, key: str) -> bool:
        """بررسی اینکه آیا درخواست مجاز است یا خیر"""
        async with asyncio.Lock():
            current = await self.redis_adapter.get(f"ratelimit:{key}")
            if current is None:
                await self.redis_adapter.set(f"ratelimit:{key}", 1, self.window_seconds)
                return True

            if int(current) >= self.max_requests:
                return False

            await self.redis_adapter.incr(f"ratelimit:{key}")
            return True

# مثال استفاده:
# limiter = RateLimiter(redis_adapter, max_requests=100, window_seconds=60)
# allowed = await limiter.is_allowed("user:123")
