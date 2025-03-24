import time
from functools import wraps


class CircuitBreaker:
    """
    پیاده‌سازی Circuit Breaker برای مدیریت خطاهای پیوسته در ارتباط با MinIO
    """

    def __init__(self, max_failures=5, reset_timeout=60):
        self.max_failures = max_failures
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"

    def _open_circuit(self):
        self.state = "OPEN"
        self.last_failure_time = time.time()

    def _half_open_circuit(self):
        self.state = "HALF-OPEN"

    def _close_circuit(self):
        self.state = "CLOSED"
        self.failure_count = 0

    def _is_timeout_expired(self):
        return (time.time() - self.last_failure_time) >= self.reset_timeout if self.last_failure_time else False

    def execute(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if self._is_timeout_expired():
                self._half_open_circuit()
            else:
                raise Exception("Circuit is OPEN. Requests are blocked.")

        try:
            result = func(*args, **kwargs)
            self._close_circuit()
            return result
        except Exception as e:
            self.failure_count += 1
            if self.failure_count >= self.max_failures:
                self._open_circuit()
            raise e

    def decorator(self, func):
        """دکوراتور برای استفاده مستقیم در توابع"""

        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.execute(func, *args, **kwargs)

        return wrapper
