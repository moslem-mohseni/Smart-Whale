from prometheus_client import Counter, Histogram
import time
from typing import Callable
from functools import wraps

# تعریف متریک‌ها
redis_operations_total = Counter(
    'redis_operations_total',
    'Total number of Redis operations',
    ['operation']
)
redis_operation_latency = Histogram(
    'redis_operation_latency_seconds',
    'Latency of Redis operations',
    ['operation']
)

def measure_redis_operation(operation: str):
    """دکوراتور برای اندازه‌گیری تعداد و زمان اجرای عملیات Redis"""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                redis_operations_total.labels(operation=operation).inc()
                redis_operation_latency.labels(operation=operation).observe(duration)
        return wrapper
    return decorator

# نصب بسته مورد نیاز:
# pip install prometheus_client
