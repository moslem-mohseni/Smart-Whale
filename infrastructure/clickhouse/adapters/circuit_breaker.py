# infrastructure/clickhouse/adapters/circuit_breaker.py
import time
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
from ..config.config import config
from ..exceptions import CircuitBreakerError, OperationalError

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """
    پیاده‌سازی Circuit Breaker برای مدیریت قطعی‌های سرویس ClickHouse

    این الگو اجازه می‌دهد سیستم در زمان خطاهای متوالی، اتصال به سرویس را موقتاً متوقف کند
    تا سرویس مقصد فرصت ریکاوری داشته باشد و از ایجاد فشار مضاعف جلوگیری شود.
    """

    def __init__(self, max_failures: int = None, reset_timeout: int = None):
        """
        مقداردهی اولیه Circuit Breaker با استفاده از تنظیمات متمرکز

        Args:
            max_failures (int, optional): حداکثر تعداد خطاها قبل از فعال شدن Circuit Breaker
            reset_timeout (int, optional): مدت زمان (ثانیه) قبل از تلاش مجدد برای اتصال

        Raises:
            OperationalError: در صورت بروز خطا در مقداردهی اولیه
        """
        try:
            # استفاده از مقادیر ارسالی یا تنظیمات متمرکز
            circuit_breaker_config = config.get_circuit_breaker_config()
            self.max_failures = max_failures if max_failures is not None else circuit_breaker_config["max_failures"]
            self.reset_timeout = reset_timeout if reset_timeout is not None else circuit_breaker_config["reset_timeout"]

            self.failure_count = 0
            self.last_failure_time = None
            self.open = False

            logger.debug(
                f"CircuitBreaker initialized with max_failures={self.max_failures}, reset_timeout={self.reset_timeout}")
        except Exception as e:
            error_msg = f"Failed to initialize CircuitBreaker: {str(e)}"
            logger.error(error_msg)
            raise OperationalError(
                message=error_msg,
                code="CB001"
            )

    def _is_reset_time_passed(self) -> bool:
        """
        بررسی می‌کند که آیا زمان بازیابی سپری شده است یا نه

        Returns:
            bool: آیا زمان بازیابی گذشته است
        """
        if self.last_failure_time is None:
            return True
        return (time.time() - self.last_failure_time) >= self.reset_timeout

    def _trip(self):
        """
        فعال‌سازی Circuit Breaker و مسدود کردن درخواست‌ها
        """
        self.open = True
        self.last_failure_time = time.time()
        logger.warning("Circuit Breaker activated: Blocking requests to ClickHouse")

    def _reset(self):
        """
        بازنشانی Circuit Breaker بعد از گذشت زمان لازم
        """
        self.open = False
        self.failure_count = 0
        self.last_failure_time = None
        logger.info("Circuit Breaker reset: Allowing requests to ClickHouse")

    def execute(self, func, *args, **kwargs):
        """
        اجرای تابع با کنترل Circuit Breaker

        Args:
            func: تابعی که باید اجرا شود
            *args: آرگومان‌های تابع
            **kwargs: آرگومان‌های کلیدی تابع

        Returns:
            نتیجه اجرای تابع در صورت موفقیت

        Raises:
            CircuitBreakerError: در صورتی که Circuit Breaker فعال باشد
            Exception: خطای اصلی در صورت بروز مشکل در اجرای تابع
        """
        if self.open:
            if self._is_reset_time_passed():
                self._reset()
            else:
                error_msg = "Circuit Breaker is open. Requests are blocked."
                logger.warning(error_msg)
                raise CircuitBreakerError(
                    message=error_msg,
                    code="CB002",
                    failure_count=self.failure_count
                )

        try:
            result = func(*args, **kwargs)
            # در صورت موفقیت، Circuit Breaker را بازنشانی می‌کنیم
            self._reset()
            return result
        except Exception as e:
            # افزایش شمارنده خطاها
            self.failure_count += 1
            if self.failure_count >= self.max_failures:
                self._trip()

            # ثبت خطا و ارسال مجدد آن
            logger.error(f"Error in CircuitBreaker.execute: {str(e)}")
            raise e

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def execute_with_retry(self, func, *args, **kwargs):
        """
        اجرای تابع با Circuit Breaker و مکانیزم Retry

        Args:
            func: تابعی که باید اجرا شود
            *args: آرگومان‌های تابع
            **kwargs: آرگومان‌های کلیدی تابع

        Returns:
            نتیجه اجرای تابع در صورت موفقیت

        Raises:
            CircuitBreakerError: در صورتی که Circuit Breaker فعال باشد
            RetryExhaustedError: در صورتی که تمام تلاش‌ها ناموفق باشند
        """
        try:
            return self.execute(func, *args, **kwargs)
        except RetryError as e:
            error_msg = "Max retry attempts reached for ClickHouse operation."
            logger.error(f"{error_msg} Original error: {str(e)}")
            from ..exceptions import RetryExhaustedError
            raise RetryExhaustedError(
                message=error_msg,
                code="CB003",
                attempts=3,  # تعداد ثابت تلاش‌ها در تنظیمات retry
                last_error=str(e)
            )
