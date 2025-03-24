import asyncio
import logging
import random
from typing import Callable, Any

logger = logging.getLogger(__name__)


class RetryMechanism:
    """
    مکانیزم مدیریت ارسال مجدد (Retry) برای پیام‌های Kafka
    """

    def __init__(self, max_retries: int = 5, base_delay: float = 0.5, max_delay: float = 10.0, jitter: bool = True):
        """
        مقداردهی اولیه RetryMechanism

        :param max_retries: حداکثر تعداد تلاش‌های مجدد
        :param base_delay: زمان انتظار اولیه برای Retry (ثانیه)
        :param max_delay: حداکثر مقدار تاخیر بین تلاش‌های مجدد
        :param jitter: اضافه کردن تغییر تصادفی به تأخیر برای جلوگیری از حملات همزمان
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter

    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        اجرای یک تابع با مکانیزم Retry

        :param func: تابعی که باید اجرا شود
        """
        attempt = 0

        while attempt < self.max_retries:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                attempt += 1
                delay = min(self.base_delay * (2 ** attempt), self.max_delay)

                if self.jitter:
                    delay += random.uniform(0, 0.1 * delay)

                logger.warning(f"Retry {attempt}/{self.max_retries} after {delay:.2f}s due to error: {e}")

                await asyncio.sleep(delay)

        logger.error(f"Max retries reached. Operation failed.")
        raise Exception("Operation failed after maximum retry attempts.")
