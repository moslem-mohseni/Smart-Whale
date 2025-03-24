import logging
import asyncio
from typing import Callable, Type

logging.basicConfig(level=logging.INFO)


class ErrorHandler:
    """
    مدیریت خطا برای جمع‌آوری داده‌ها، شامل تلاش مجدد و ثبت لاگ.
    """

    def __init__(self, max_retries: int = 3, retry_delay: float = 2.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    async def execute_with_retry(self, func: Callable, *args, **kwargs):
        """
        اجرای یک تابع با مکانیزم تلاش مجدد در صورت بروز خطا.
        """
        retries = 0
        while retries < self.max_retries:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                retries += 1
                logging.warning(f"⚠ Attempt {retries}/{self.max_retries} failed: {e}")
                if retries < self.max_retries:
                    await asyncio.sleep(self.retry_delay)
                else:
                    logging.error(f"❌ All {self.max_retries} attempts failed for {func.__name__}")
                    raise

    def log_error(self, error: Exception, context: str = ""):
        """
        ثبت خطا در لاگ همراه با متن زمینه‌ای.
        """
        logging.error(f"❌ Error in {context}: {error}")


if __name__ == "__main__":
    async def sample_task():
        raise ValueError("Test error")


    handler = ErrorHandler(max_retries=3, retry_delay=1)

    asyncio.run(handler.execute_with_retry(sample_task))