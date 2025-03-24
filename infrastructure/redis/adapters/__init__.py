"""
ماژول Adapters شامل تمام کلاس‌ها و ابزارهای مرتبط با ارتباط با Redis است.
- `redis_adapter.py`: مدیریت اتصال و عملیات CRUD
- `circuit_breaker.py`: مدیریت خطاهای اتصال
- `connection_pool.py`: مدیریت Connection Pooling
- `retry_mechanism.py`: اجرای مجدد درخواست‌های ناموفق
"""

from .redis_adapter import RedisAdapter
from .circuit_breaker import CircuitBreaker
from .connection_pool import RedisConnectionPool
from .retry_mechanism import retry_async

__all__ = [
    "RedisAdapter",
    "CircuitBreaker",
    "RedisConnectionPool",
    "retry_async"
]
