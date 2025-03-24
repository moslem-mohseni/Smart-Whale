import logging
from typing import Dict
from ..adapters.redis_adapter import RedisAdapter

logger = logging.getLogger(__name__)

class HealthCheck:
    """
    بررسی سلامت و دسترس‌پذیری Redis
    """
    def __init__(self, redis_adapter: RedisAdapter):
        self.redis_adapter = redis_adapter

    async def check(self) -> Dict[str, str]:
        """اجرای تست سلامت برای بررسی دسترس‌پذیری Redis"""
        status = {"redis": "unhealthy"}
        try:
            await self.redis_adapter.set("health_check", "ok", ttl=5)
            value = await self.redis_adapter.get("health_check")
            if value == "ok":
                status["redis"] = "healthy"
        except Exception as e:
            logger.error(f"Redis health check failed: {str(e)}")
        return status

# مثال استفاده:
# health_checker = HealthCheck(redis_adapter)
# status = await health_checker.check()
# print(status)
