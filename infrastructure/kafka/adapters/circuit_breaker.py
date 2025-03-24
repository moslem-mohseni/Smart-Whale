import time
import logging
from typing import Callable

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """
    پیاده‌سازی Circuit Breaker برای مدیریت خطاهای Kafka
    """

    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 60, half_open_attempts: int = 2):
        """
        مقداردهی اولیه Circuit Breaker

        :param failure_threshold: تعداد شکست‌های متوالی قبل از باز شدن Circuit Breaker
        :param reset_timeout: مدت زمان (ثانیه) که Circuit Breaker در حالت OPEN باقی می‌ماند
        :param half_open_attempts: تعداد درخواست‌های تست در حالت HALF-OPEN
        """
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.half_open_attempts = half_open_attempts

        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"
        self.success_attempts = 0

    def allow_request(self) -> bool:
        """
        بررسی می‌کند که آیا درخواست مجاز است یا خیر.
        """
        if self.state == "OPEN":
            if (time.time() - self.last_failure_time) > self.reset_timeout:
                self.state = "HALF-OPEN"
                logger.info("Circuit Breaker moved to HALF-OPEN state.")
                return True
            return False

        return True

    def record_success(self):
        """
        ثبت موفقیت‌آمیز بودن یک درخواست
        """
        if self.state == "HALF-OPEN":
            self.success_attempts += 1
            if self.success_attempts >= self.half_open_attempts:
                self._reset()
                logger.info("Circuit Breaker moved to CLOSED state.")

    def record_failure(self):
        """
        ثبت شکست یک درخواست
        """
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"Circuit Breaker moved to OPEN state. Waiting {self.reset_timeout} seconds.")

    def _reset(self):
        """
        بازنشانی Circuit Breaker به حالت نرمال
        """
        self.failure_count = 0
        self.success_attempts = 0
        self.state = "CLOSED"

    def execute(self, func: Callable, *args, **kwargs):
        """
        اجرای یک تابع تحت کنترل Circuit Breaker

        :param func: تابعی که باید اجرا شود
        """
        if not self.allow_request():
            raise Exception("Circuit Breaker is OPEN. Request denied.")

        try:
            result = func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure()
            logger.error(f"Error in Circuit Breaker execution: {str(e)}")
            raise
