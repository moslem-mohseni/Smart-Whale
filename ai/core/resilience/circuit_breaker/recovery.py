import time
import logging
from prometheus_client import Counter
from ai.core.resilience.circuit_breaker.state_manager import CircuitBreakerStateManager


class CircuitBreakerRecovery:
    def __init__(self, state_manager: CircuitBreakerStateManager, test_function, recovery_attempts=3):
        """
        مدیریت بازیابی خودکار Circuit Breaker پس از قطع شدن درخواست‌ها
        :param state_manager: نمونه‌ای از CircuitBreakerStateManager برای دریافت وضعیت
        :param test_function: تابع تست برای بررسی سلامت سرویس
        :param recovery_attempts: تعداد تلاش‌های مجدد برای بازیابی
        """
        self.state_manager = state_manager
        self.test_function = test_function
        self.recovery_attempts = recovery_attempts
        self.logger = logging.getLogger("CircuitBreakerRecovery")

        # متریک Prometheus برای شمارش تلاش‌های بازیابی
        self.recovery_attempts_metric = Counter("circuit_breaker_recovery_attempts", "Total recovery attempts for circuit breaker")

    async def attempt_recovery(self):
        """
        تلاش برای بازیابی سرویس و بازگرداندن Circuit Breaker به وضعیت CLOSED در صورت موفقیت
        """
        state_data = await self.state_manager.get_state()
        if state_data["state"] != "OPEN":
            return  # اگر Circuit Breaker در وضعیت OPEN نیست، نیازی به بازیابی نیست

        elapsed_time = time.time() - state_data["last_failure_time"]
        if elapsed_time < 30:  # اگر هنوز زمان بازیابی نرسیده، منتظر بمانیم
            self.logger.info("⏳ زمان بازیابی هنوز نرسیده، تلاش مجدد بعداً انجام خواهد شد.")
            return

        self.logger.info("🔄 Circuit Breaker در حالت HALF-OPEN قرار گرفت، تلاش برای بازیابی...")

        for attempt in range(1, self.recovery_attempts + 1):
            self.logger.info(f"🚀 تلاش {attempt} برای بازیابی سرویس...")
            self.recovery_attempts_metric.inc()

            try:
                if await self.test_function():
                    self.logger.info("✅ بازیابی موفقیت‌آمیز بود! Circuit Breaker به حالت CLOSED بازمی‌گردد.")
                    await self.state_manager.save_state("CLOSED")
                    return
            except Exception as e:
                self.logger.warning(f"⚠️ تلاش {attempt} برای بازیابی ناموفق بود: {e}")

            time.sleep(5)  # تأخیر قبل از تلاش مجدد

        self.logger.error("❌ تمامی تلاش‌ها برای بازیابی ناموفق بود، Circuit Breaker در حالت OPEN باقی می‌ماند.")
