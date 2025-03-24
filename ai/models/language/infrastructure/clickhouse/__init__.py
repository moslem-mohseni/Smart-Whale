"""
ماژول `clickhouse/` وظیفه‌ی مدیریت ذخیره‌سازی و پردازش داده‌های تحلیلی پردازش زبان را بر عهده دارد.

📌 اجزای اصلی این ماژول:
- `clickhouse_adapter.py` → مدیریت ارتباط با ClickHouse و اجرای کوئری‌های تحلیلی
"""

from .clickhouse_adapter import ClickHouseDB
from infrastructure.clickhouse.adapters.clickhouse_adapter import ClickHouseAdapter
from infrastructure.clickhouse.optimization.cache_manager import CacheManager
from infrastructure.clickhouse.optimization.query_optimizer import QueryOptimizer

# مقداردهی اولیه سرویس‌های ClickHouse
clickhouse_adapter = ClickHouseAdapter()
cache_manager = CacheManager()
query_optimizer = QueryOptimizer()

# مقداردهی اولیه ClickHouseDB
clickhouse_db = ClickHouseDB(clickhouse_adapter, cache_manager, query_optimizer)

__all__ = [
    "clickhouse_db",
    "ClickHouseDB",
]
