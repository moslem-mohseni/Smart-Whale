from dataclasses import dataclass
from typing import Any, Optional, Dict
from datetime import datetime, timedelta

@dataclass
class CacheItem:
    """
    مدل داده‌ای برای آیتم‌های کش شده در Redis
    """
    key: str
    value: Any
    ttl: Optional[int] = None  # زمان انقضا به ثانیه
    created_at: datetime = None
    metadata: Optional[Dict] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

    @property
    def is_expired(self) -> bool:
        """بررسی وضعیت انقضای آیتم کش"""
        if self.ttl is None:
            return False
        expiry_time = self.created_at + timedelta(seconds=self.ttl)
        return datetime.now() > expiry_time

@dataclass
class CacheNamespace:
    """
    فضای نام برای گروه‌بندی کلیدهای مرتبط در Redis
    """
    name: str
    default_ttl: Optional[int] = None
    max_size: Optional[int] = None
    eviction_policy: str = 'lru'  # نحوه حذف داده‌ها در صورت پر شدن فضا
