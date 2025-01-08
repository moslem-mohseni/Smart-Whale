# infrastructure/timescaledb/domain/models.py

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

@dataclass
class TimeSeriesData:
    """مدل پایه برای داده‌های سری زمانی"""
    id: str
    timestamp: datetime
    value: float
    metadata: dict = None


@dataclass
class TableSchema:
    """مدل برای تعریف ساختار جداول

    این کلاس برای تعریف ساختار جداول استفاده می‌شود و شامل نام جدول و
    مشخصات ستون‌های آن است.
    """
    name: str
    columns: Dict[str, str]
    time_column: Optional[str] = None
    indexes: Optional[Dict[str, str]] = None
    constraints: Optional[Dict[str, str]] = None

    def __post_init__(self):
        """اعتبارسنجی پس از مقداردهی اولیه"""
        if not self.name:
            raise ValueError("Table name cannot be empty")
        if not self.columns:
            raise ValueError("Columns cannot be empty")
        if self.time_column and self.time_column not in self.columns:
            raise ValueError(f"Time column {self.time_column} not found in columns")

    def get_creation_sql(self) -> str:
        """تولید دستور SQL برای ایجاد جدول

        Returns:
            str: دستور SQL
        """
        # تعریف ستون‌ها
        column_defs = [f"{name} {dtype}" for name, dtype in self.columns.items()]

        # اضافه کردن محدودیت‌ها
        if self.constraints:
            column_defs.extend(self.constraints.values())

        # ساخت دستور SQL نهایی
        sql = f"""
            CREATE TABLE IF NOT EXISTS {self.name} (
                {', '.join(column_defs)}
            )
        """
        return sql

    def get_index_creation_sql(self) -> list[str]:
        """تولید دستورات SQL برای ایجاد ایندکس‌ها

        Returns:
            list[str]: لیست دستورات SQL
        """
        if not self.indexes:
            return []

        return [
            f"CREATE INDEX IF NOT EXISTS {name} ON {self.name} {definition}"
            for name, definition in self.indexes.items()
        ]