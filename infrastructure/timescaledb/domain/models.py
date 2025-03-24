from dataclasses import dataclass
from datetime import datetime
from typing import Dict


@dataclass
class TimeSeriesData:
    """مدل داده‌های سری‌زمانی"""
    id: int
    timestamp: datetime
    value: float
    metadata: Dict[str, str]


@dataclass
class TableSchema:
    """مدل طرح جداول در پایگاه داده"""
    name: str  # نام جدول
    columns: Dict[str, str]  # ساختار جدول {نام ستون: نوع داده}
    time_column: str  # ستون زمانی برای Hypertable
    indexes: Dict[str, str] = None  # ایندکس‌ها (اختیاری)
