"""
ماژول `timescaledb/` وظیفه‌ی مدیریت داده‌های سری‌زمانی در TimescaleDB را بر عهده دارد.

📌 اجزای اصلی این ماژول:
- `timescaledb_adapter.py` → مدیریت ارتباط با TimescaleDB و انجام عملیات سری‌زمانی
- `metrics_handler.py` → دریافت متریک‌های ذخیره‌سازی و جستجو
"""

from .timescaledb_adapter import TimescaleDBAdapter
from .metrics_handler import MetricsHandler
from infrastructure.timescaledb.service.database_service import DatabaseService

# مقداردهی اولیه DatabaseService
database_service = DatabaseService()

# مقداردهی اولیه TimescaleDBAdapter و MetricsHandler
timescaledb_adapter = TimescaleDBAdapter(database_service)
metrics_handler = MetricsHandler(database_service)

__all__ = [
    "timescaledb_adapter",
    "metrics_handler",
    "TimescaleDBAdapter",
    "MetricsHandler",
]
