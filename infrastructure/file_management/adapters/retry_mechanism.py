import time
from functools import wraps


class RetryMechanism:
    """
    پیاده‌سازی مکانیزم Retry برای عملیات ناموفق
    """

    def __init__(self, max_attempts=5, min_wait=2, max_wait=10):
        self.max_attempts = max_attempts
        self.min_wait = min_wait
        self.max_wait = max_wait

    def retry(self, func):
        """دکوراتور برای اجرای مجدد عملیات در صورت شکست"""

        @wraps(func)
        async def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < self.max_attempts:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    wait_time = min(self.max_wait, self.min_wait * (2 ** attempts))
                    time.sleep(wait_time)
            raise Exception("Max retry attempts reached")

        return wrapper
