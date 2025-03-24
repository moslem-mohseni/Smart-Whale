import time
import logging
from prometheus_client import Gauge


class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_time=30):
        """
        مدیریت Circuit Breaker برای کنترل درخواست‌های ناموفق
        :param failure_threshold: تعداد خطاهای مجاز قبل از فعال شدن Circuit Breaker
        :param recovery_time: زمان انتظار (ثانیه) قبل از بررسی مجدد درخواست‌ها در حالت HALF-OPEN
        """
        self.failure_threshold = failure_threshold
        self.recovery_time = recovery_time
        self.failure_count = 0
        self.state = "CLOSED"  # وضعیت اولیه: عادی
        self.last_failure_time = None
        self.logger = logging.getLogger("CircuitBreaker")

        # متریک‌های Prometheus
        self.breaker_state = Gauge("circuit_breaker_state", "Current state of the circuit breaker", ["state"])
        self.update_prometheus_metrics()

    def update_prometheus_metrics(self):
        """ به‌روزرسانی وضعیت Circuit Breaker در Prometheus """
        self.breaker_state.labels(state=self.state).set(1)

    def allow_request(self):
        """
        بررسی اینکه آیا درخواست جدید مجاز است یا خیر
        :return: True اگر مجاز باشد، False در صورت فعال بودن Circuit Breaker
        """
        if self.state == "OPEN":
            elapsed_time = time.time() - self.last_failure_time
            if elapsed_time >= self.recovery_time:
                self.state = "HALF-OPEN"
                self.logger.info("🔄 Circuit Breaker به حالت HALF-OPEN تغییر وضعیت داد.")
                self.update_prometheus_metrics()
            else:
                self.logger.warning("⚠️ Circuit Breaker در وضعیت OPEN است. درخواست مجاز نیست.")
                return False

        return True

    def record_success(self):
        """ ثبت یک درخواست موفق و بازگرداندن Circuit Breaker به حالت عادی در صورت نیاز """
        if self.state == "HALF-OPEN":
            self.logger.info("✅ Circuit Breaker به وضعیت CLOSED بازگشت.")
            self.state = "CLOSED"
        self.failure_count = 0
        self.update_prometheus_metrics()

    def record_failure(self):
        """ ثبت یک درخواست ناموفق و بررسی فعال شدن Circuit Breaker """
        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            self.last_failure_time = time.time()
            self.logger.error("🚨 Circuit Breaker فعال شد! درخواست‌ها مسدود می‌شوند.")
        self.update_prometheus_metrics()
