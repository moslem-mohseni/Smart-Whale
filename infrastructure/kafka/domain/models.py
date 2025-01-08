# infrastructure / kafka / domain / models.py
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Dict


@dataclass
class Message:
    """
    کلاس مدل برای پیام‌های کافکا

    این کلاس ساختار پایه یک پیام را تعریف می‌کند که شامل محتوای پیام،
    زمان ایجاد و متادیتای اضافی است.
    """
    topic: str
    content: Any
    timestamp: datetime = None
    metadata: Optional[Dict] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class TopicConfig:
    """
    تنظیمات یک موضوع در کافکا

    این کلاس پارامترهای پیکربندی یک موضوع را نگهداری می‌کند،
    مانند تعداد پارتیشن‌ها و فاکتور تکرار.
    """
    name: str
    partitions: int = 1
    replication_factor: int = 1
    configs: Dict[str, Any] = None
