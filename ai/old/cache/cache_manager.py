from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import json
import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """تنظیمات کش"""
    max_size: int = 1000  # حداکثر تعداد آیتم‌ها
    ttl: int = 3600  # زمان اعتبار به ثانیه
    cleanup_interval: int = 300  # فاصله پاکسازی به ثانیه


@dataclass
class CacheItem:
    """آیتم کش"""
    key: str
    value: Any
    expires_at: datetime
    metadata: Optional[Dict[str, Any]] = None


class CacheManager:
    """مدیریت کش"""

    def __init__(self, config: CacheConfig = CacheConfig()):
        self.config = config
        self._cache: Dict[str, CacheItem] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None

    async def start(self):
        """شروع پاکسازی خودکار"""
        if not self._cleanup_task:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self):
        """توقف پاکسازی خودکار"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            await asyncio.gather(self._cleanup_task, return_exceptions=True)
            self._cleanup_task = None

    async def get(self, key: str) -> Optional[Any]:
        """دریافت مقدار از کش"""
        async with self._lock:
            if item := self._cache.get(key):
                if datetime.now() < item.expires_at:
                    return item.value
                del self._cache[key]
        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """ذخیره مقدار در کش"""
        expires_at = datetime.now() + timedelta(seconds=ttl or self.config.ttl)
        item = CacheItem(key=key, value=value, expires_at=expires_at)

        async with self._lock:
            if len(self._cache) >= self.config.max_size:
                await self._evict_one()
            self._cache[key] = item

    async def delete(self, key: str) -> None:
        """حذف مقدار از کش"""
        async with self._lock:
            self._cache.pop(key, None)

    async def clear(self) -> None:
        """پاکسازی کامل کش"""
        async with self._lock:
            self._cache.clear()

    async def _cleanup_loop(self):
        """پاکسازی خودکار آیتم‌های منقضی"""
        while True:
            try:
                await asyncio.sleep(self.config.cleanup_interval)
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")

    async def _cleanup_expired(self):
        """پاکسازی آیتم‌های منقضی"""
        now = datetime.now()
        async with self._lock:
            expired = [k for k, v in self._cache.items() if v.expires_at <= now]
            for key in expired:
                del self._cache[key]

    async def _evict_one(self):
        """حذف یک آیتم برای آزادسازی فضا"""
        if not self._cache:
            return

        # حذف قدیمی‌ترین آیتم
        oldest_key = min(self._cache.items(), key=lambda x: x[1].expires_at)[0]
        del self._cache[oldest_key]