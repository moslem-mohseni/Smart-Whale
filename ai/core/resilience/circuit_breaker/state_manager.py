import time
import logging
import aioredis
from prometheus_client import Counter
from infrastructure.redis.service.cache_service import CacheService


class CircuitBreakerStateManager:
    def __init__(self, redis_client: CacheService, service_name: str):
        """
        مدیریت وضعیت Circuit Breaker برای سرویس‌های مختلف
        :param redis_client: کلاینت Redis برای ذخیره وضعیت Circuit Breaker
        :param service_name: نام سرویس مرتبط با Circuit Breaker
        """
        self.redis = redis_client
        self.service_name = service_name
        self.logger = logging.getLogger(f"CircuitBreaker-{service_name}")

        # متریک Prometheus برای شمارش تعداد تغییر وضعیت Circuit Breaker
        self.state_changes = Counter(f"{service_name}_circuit_breaker_state_changes",
                                     f"Total circuit breaker state changes for {service_name}")

    async def save_state(self, state: str, last_failure_time: float = None):
        """
        ذخیره وضعیت Circuit Breaker در Redis
        :param state: وضعیت فعلی (OPEN، CLOSED، HALF-OPEN)
        :param last_failure_time: زمان آخرین شکست برای حالت OPEN
        """
        data = {
            "state": state,
            "last_failure_time": last_failure_time if last_failure_time else time.time()
        }
        await self.redis.hset(f"circuit_breaker:{self.service_name}", mapping=data)
        self.state_changes.inc()  # افزایش شمارنده تغییر وضعیت
        self.logger.info(f"✅ وضعیت Circuit Breaker برای {self.service_name} ذخیره شد: {state}")

    async def get_state(self):
        """
        دریافت وضعیت فعلی Circuit Breaker از Redis
        :return: دیکشنری شامل وضعیت و زمان آخرین شکست
        """
        data = await self.redis.hgetall(f"circuit_breaker:{self.service_name}")
        if not data:
            return {"state": "CLOSED", "last_failure_time": None}
        return {
            "state": data.get("state", "CLOSED"),
            "last_failure_time": float(data.get("last_failure_time", 0))
        }

    async def reset_state(self):
        """
        بازنشانی وضعیت Circuit Breaker به حالت CLOSED
        """
        await self.save_state("CLOSED")
        self.logger.info(f"🔄 وضعیت Circuit Breaker برای {self.service_name} به حالت CLOSED بازنشانی شد.")
