# infrastructure/clickhouse/adapters/retry_mechanism.py
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from clickhouse_driver.errors import Error as ClickHouseDriverError
from .circuit_breaker import CircuitBreaker
from ..config.config import config
from ..exceptions import RetryExhaustedError, OperationalError

logger = logging.getLogger(__name__)


class RetryHandler:
    """
    مدیریت Retry برای عملیات ClickHouse

    این کلاس امکان تلاش مجدد در عملیات‌های ClickHouse را با استفاده از کتابخانه tenacity
    فراهم می‌کند و می‌تواند در ترکیب با CircuitBreaker استفاده شود.
    """

    def __init__(self, max_attempts: int = None, min_wait: int = None, max_wait: int = None):
        """
        مقداردهی اولیه RetryHandler با استفاده از تنظیمات متمرکز

        Args:
            max_attempts (int, optional): حداکثر تعداد تلاش مجدد
            min_wait (int, optional): حداقل زمان انتظار بین تلاش‌ها (ثانیه)
            max_wait (int, optional): حداکثر زمان انتظار بین تلاش‌ها (ثانیه)

        Raises:
            OperationalError: در صورت بروز خطا در مقداردهی اولیه
        """
        try:
            # استفاده از مقادیر ارسالی یا تنظیمات متمرکز
            retry_config = config.get_retry_config()
            self.max_attempts = max_attempts if max_attempts is not None else retry_config["max_attempts"]
            self.min_wait = min_wait if min_wait is not None else retry_config["min_wait"]
            self.max_wait = max_wait if max_wait is not None else retry_config["max_wait"]

            # ایجاد یک Circuit Breaker با تنظیمات پیش‌فرض
            self.circuit_breaker = CircuitBreaker()

            logger.debug(
                f"RetryHandler initialized with max_attempts={self.max_attempts}, min_wait={self.min_wait}, max_wait={self.max_wait}")

        except Exception as e:
            error_msg = f"Failed to initialize RetryHandler: {str(e)}"
            logger.error(error_msg)
            raise OperationalError(
                message=error_msg,
                code="RETRY001"
            )

    def execute_with_retry(self, func, *args, **kwargs):
        """
        اجرای تابع با مکانیزم Retry دینامیک بر اساس تنظیمات

        Args:
            func: تابعی که باید اجرا شود
            *args: آرگومان‌های تابع
            **kwargs: آرگومان‌های کلیدی تابع

        Returns:
            نتیجه اجرای تابع در صورت موفقیت

        Raises:
            RetryExhaustedError: در صورتی که همه تلاش‌ها ناموفق باشند
            CircuitBreakerError: در صورتی که Circuit Breaker فعال باشد
            ClickHouseDriverError: در صورت بروز خطای ClickHouse
            Exception: سایر خطاها
        """
        # ایجاد دکوراتور retry دینامیک بر اساس تنظیمات
        retry_decorator = retry(
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_exponential(multiplier=1, min=self.min_wait, max=self.max_wait),
            retry=retry_if_exception_type(ClickHouseDriverError),
        )

        # اعمال دکوراتور به یک تابع درونی
        @retry_decorator
        def _execute_with_retry():
            try:
                # استفاده از Circuit Breaker برای اجرای تابع
                return self.circuit_breaker.execute(func, *args, **kwargs)
            except ClickHouseDriverError as e:
                logger.warning(f"Retry attempt failed: {str(e)}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error in retry attempt: {str(e)}")
                raise

        try:
            # اجرای تابع با retry
            return _execute_with_retry()
        except Exception as e:
            if "RetryError" in str(type(e)):
                # تبدیل خطای tenacity به خطای سفارشی ما
                error_msg = f"Max retry attempts ({self.max_attempts}) reached for ClickHouse operation."
                logger.error(error_msg)
                raise RetryExhaustedError(
                    message=error_msg,
                    code="RETRY002",
                    attempts=self.max_attempts,
                    last_error=str(e)
                )
            else:
                # سایر خطاها را مستقیماً منتقل می‌کنیم
                raise
