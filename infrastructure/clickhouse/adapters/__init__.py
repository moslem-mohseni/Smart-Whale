# infrastructure/clickhouse/adapters/__init__.py
"""
ماژول آداپتورهای ClickHouse

این ماژول شامل کلاس‌ها و توابع لازم برای ارتباط با ClickHouse است:
- ConnectionPool: مدیریت اتصالات به ClickHouse
- CircuitBreaker: مدیریت قطعی‌های سرویس
- RetryHandler: مکانیزم تلاش مجدد
- LoadBalancer: توزیع بار بین سرورها
- ClickHouseAdapter: آداپتور مرکزی برای اجرای کوئری‌ها
"""

import logging
from ..config.config import config
from ..exceptions import ConnectionError, OperationalError
from .connection_pool import ClickHouseConnectionPool
from .circuit_breaker import CircuitBreaker
from .retry_mechanism import RetryHandler
from .load_balancer import ClickHouseLoadBalancer
from .clickhouse_adapter import ClickHouseAdapter

logger = logging.getLogger(__name__)

logger.info("Initializing ClickHouse Adapters Module...")

__all__ = [
    "ClickHouseConnectionPool",
    "CircuitBreaker",
    "RetryHandler",
    "ClickHouseLoadBalancer",
    "ClickHouseAdapter",
    "ConnectionError",
    "create_adapter"
]


def create_adapter(custom_config=None) -> ClickHouseAdapter:
    """
    ایجاد یک نمونه از آداپتور ClickHouse با تنظیمات مناسب

    این تابع یک راه ساده برای ایجاد آداپتور با تنظیمات پیش‌فرض یا سفارشی فراهم می‌کند.

    Args:
        custom_config: تنظیمات سفارشی (اختیاری)

    Returns:
        ClickHouseAdapter: یک نمونه از آداپتور ClickHouse آماده استفاده

    Raises:
        ConnectionError: در صورت بروز خطا در اتصال اولیه به ClickHouse
        OperationalError: در صورت بروز خطا در مقداردهی اولیه آداپتور
    """
    try:
        logger.debug("Creating new ClickHouseAdapter instance")
        adapter = ClickHouseAdapter(custom_config or config)
        return adapter
    except Exception as e:
        error_msg = f"Failed to create ClickHouseAdapter: {str(e)}"
        logger.error(error_msg)
        raise OperationalError(
            message=error_msg,
            code="ADP001"
        )
