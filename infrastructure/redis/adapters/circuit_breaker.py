import asyncio
from datetime import datetime, timedelta
from typing import Optional


class CircuitBreaker:
    """
    Circuit Breaker برای مدیریت شکست‌های اتصال به Redis
    """

    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 60):
        self.failure_threshold = failure_threshold  # تعداد خطاهای مجاز قبل از تغییر به حالت OPEN
        self.reset_timeout = reset_timeout  # مدت زمان انتظار برای ریست شدن
        self.failure_count = 0  # تعداد شکست‌های ثبت‌شده
        self.last_failure_time: Optional[datetime] = None  # زمان آخرین شکست
        self.state = "CLOSED"  # وضعیت اولیه Circuit Breaker
        self.lock = asyncio.Lock()  # برای مدیریت رقابت همزمانی

    async def __aenter__(self):
        """ بررسی وضعیت Circuit Breaker قبل از اجرای درخواست """
        async with self.lock:
            if self.state == "OPEN":
                if self.last_failure_time and datetime.now() - self.last_failure_time > timedelta(
                        seconds=self.reset_timeout):
                    self.state = "HALF_OPEN"
                else:
                    raise ConnectionError("Circuit breaker is OPEN. Requests are blocked.")
            return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """ بررسی موفقیت‌آمیز بودن عملیات و بروزرسانی وضعیت Circuit Breaker """
        if exc_type is not None:
            await self.record_failure()
        else:
            await self.record_success()

    async def record_failure(self):
        """ ثبت یک شکست در درخواست و بررسی نیاز به فعال‌سازی Circuit Breaker """
        async with self.lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"

    async def record_success(self):
        """ در صورت موفقیت‌آمیز بودن درخواست، مقداردهی مجدد Circuit Breaker """
        async with self.lock:
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
            self.failure_count = 0
