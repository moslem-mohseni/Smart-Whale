import time
import functools
from typing import Callable
from ..config.config import config


class RetryMechanism:
    """مدیریت تلاش مجدد (Retry) برای درخواست‌های ناموفق"""

    def __init__(self, max_attempts: int = None, delay: float = None):
        """
        مقداردهی اولیه Retry Mechanism
        :param max_attempts: حداکثر تعداد تلاش مجدد
        :param delay: فاصله زمانی اولیه بین تلاش‌ها (به صورت نمایی افزایش می‌یابد)
        """
        self.max_attempts = max_attempts or int(config.MILVUS_RETRY_ATTEMPTS)
        self.delay = delay or float(getattr(config, 'MILVUS_RETRY_DELAY', 1.0))

    def retry(self, func: Callable):
        """دکوراتور برای تلاش مجدد روی توابع ناموفق"""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            while attempt < self.max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    wait_time = self.delay * (2 ** (attempt - 1))  # بک‌آف نمایی
                    print(f"⚠️ تلاش {attempt}/{self.max_attempts} برای اجرای `{func.__name__}` ناموفق بود. خطا: {e}")
                    print(f"⏳ انتظار برای {wait_time:.2f} ثانیه قبل از تلاش مجدد...")
                    time.sleep(wait_time)
            print(f"❌ شکست در اجرای `{func.__name__}` بعد از {self.max_attempts} تلاش.")
            raise Exception(f"Maximum retry attempts reached for `{func.__name__}`")

        return wrapper
