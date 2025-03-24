# infrastructure/clickhouse/exceptions/__init__.py
"""
ماژول خطاهای سفارشی ClickHouse

این ماژول شامل تمامی کلاس‌های خطای سفارشی برای ماژول ClickHouse است.
"""

from .base import ClickHouseBaseError
from .connection_errors import (
    ConnectionError,
    PoolExhaustedError,
    ConnectionTimeoutError,
    AuthenticationError
)
from .query_errors import (
    QueryError,
    QuerySyntaxError,
    QueryExecutionTimeoutError,
    QueryCancellationError,
    DataTypeError
)
from .security_errors import (
    SecurityError,
    EncryptionError,
    TokenError,
    PermissionDeniedError
)
from .operational_errors import (
    OperationalError,
    CircuitBreakerError,
    RetryExhaustedError,
    BackupError,
    DataManagementError
)

# صادر کردن تمامی کلاس‌های خطا
__all__ = [
    # کلاس پایه
    "ClickHouseBaseError",

    # خطاهای اتصال
    "ConnectionError",
    "PoolExhaustedError",
    "ConnectionTimeoutError",
    "AuthenticationError",

    # خطاهای کوئری
    "QueryError",
    "QuerySyntaxError",
    "QueryExecutionTimeoutError",
    "QueryCancellationError",
    "DataTypeError",

    # خطاهای امنیتی
    "SecurityError",
    "EncryptionError",
    "TokenError",
    "PermissionDeniedError",

    # خطاهای عملیاتی
    "OperationalError",
    "CircuitBreakerError",
    "RetryExhaustedError",
    "BackupError",
    "DataManagementError"
]
