# infrastructure/clickhouse/domain/models.py

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, List


@dataclass
class AnalyticsEvent:
    """
    مدل پایه برای رویدادهای تحلیلی

    این کلاس برای نگهداری داده‌های رویدادهایی استفاده می‌شود که نیاز به تحلیل دارند.
    ClickHouse به طور خاص برای پردازش چنین داده‌هایی بهینه شده است.
    """
    event_id: str
    event_type: str
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AnalyticsQuery:
    """
    مدل پرس‌وجوهای تحلیلی

    این کلاس برای ساخت‌یافته کردن پرس‌وجوهای تحلیلی استفاده می‌شود.
    شامل پارامترهای مورد نیاز برای اجرای تحلیل‌های پیچیده است.
    """
    dimensions: List[str]  # ابعاد مورد نظر برای گروه‌بندی
    metrics: List[str]  # معیارهای محاسباتی
    filters: Optional[Dict[str, Any]] = None  # فیلترهای اعمالی
    time_range: Optional[tuple[datetime, datetime]] = None  # بازه زمانی
    limit: Optional[int] = None  # محدودیت تعداد نتایج
    order_by: Optional[List[str]] = None  # ترتیب نتایج


@dataclass
class AnalyticsResult:
    """
    مدل نتایج تحلیلی

    این کلاس برای نگهداری و سازماندهی نتایج پرس‌وجوهای تحلیلی استفاده می‌شود.
    امکان دسته‌بندی و پردازش بیشتر نتایج را فراهم می‌کند.
    """
    query: AnalyticsQuery
    data: List[Dict[str, Any]]
    total_count: int
    execution_time: float  # زمان اجرای پرس‌وجو به ثانیه
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TableSchema:
    """
    مدل طرح جدول در ClickHouse

    این کلاس برای تعریف ساختار جداول در ClickHouse استفاده می‌شود.
    شامل تنظیمات خاص ClickHouse مانند موتور جدول و سیاست‌های بهینه‌سازی است.
    """
    name: str
    columns: Dict[str, str]  # نام ستون: نوع داده
    engine: str = 'MergeTree()'
    partition_by: Optional[str] = None
    order_by: Optional[List[str]] = None
    sample_by: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None

    def get_create_table_sql(self) -> str:
        """تولید دستور SQL برای ایجاد جدول"""
        columns_sql = ", ".join(f"{name} {type_}" for name, type_ in self.columns.items())
        sql = f"CREATE TABLE IF NOT EXISTS {self.name} ({columns_sql}) ENGINE = {self.engine}"

        if self.partition_by:
            sql += f" PARTITION BY {self.partition_by}"
        if self.order_by:
            sql += f" ORDER BY ({', '.join(self.order_by)})"
        if self.sample_by:
            sql += f" SAMPLE BY {self.sample_by}"
        if self.settings:
            settings_sql = ", ".join(f"{k}={v}" for k, v in self.settings.items())
            sql += f" SETTINGS {settings_sql}"

        return sql