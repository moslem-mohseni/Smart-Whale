# tests/unit/infrastructure/test_interfaces.py

import pytest
from typing import Any, List, Optional, Dict
import fnmatch
from datetime import datetime, timedelta
from infrastructure.interfaces import (
    StorageInterface,
    CachingInterface,
    MessagingInterface,
    InfrastructureError,
    ConnectionError
)


# ساخت پیاده‌سازی‌های Mock برای تست اینترفیس‌ها


class MockStorage(StorageInterface):
    """پیاده‌سازی Mock برای تست StorageInterface"""

    def __init__(self):
        self._connected = False
        self._available = True
        self._tables = {}
        self._transaction = False
        self._query_results = []

    async def connect(self) -> None:
        if not self._available:
            raise ConnectionError("Storage is not available")
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def is_connected(self) -> bool:
        return self._connected

    async def ping(self) -> bool:
        return self._available

    async def execute(self, query: str, params: Optional[List[Any]] = None) -> List[Any]:
        """شبیه‌سازی اجرای پرس‌وجو"""
        if not self._connected:
            raise ConnectionError("Not connected")
        return self._query_results

    async def execute_many(self, query: str, params_list: List[List[Any]]) -> None:
        """شبیه‌سازی اجرای چندین پرس‌وجو"""
        if not self._connected:
            raise ConnectionError("Not connected")
        for params in params_list:
            await self.execute(query, params)

    async def begin_transaction(self) -> None:
        """شروع تراکنش"""
        if self._transaction:
            raise ValueError("Transaction already started")
        self._transaction = True

    async def commit(self) -> None:
        """تأیید تراکنش"""
        if not self._transaction:
            raise ValueError("No transaction in progress")
        self._transaction = False

    async def rollback(self) -> None:
        """برگشت تراکنش"""
        if not self._transaction:
            raise ValueError("No transaction in progress")
        self._transaction = False

    async def create_table(self, table_name: str, schema: dict) -> None:
        """ایجاد جدول جدید"""
        if table_name in self._tables:
            raise ValueError(f"Table {table_name} already exists")
        self._tables[table_name] = schema

    async def create_hypertable(self, table_name: str, time_column: str) -> None:
        """تبدیل جدول به hypertable"""
        if table_name not in self._tables:
            raise ValueError(f"Table {table_name} does not exist")
        # در mock نیازی به پیاده‌سازی واقعی نیست

    def set_available(self, available: bool):
        """تنظیم وضعیت در دسترس بودن برای تست"""
        self._available = available

    def set_query_results(self, results: List[Any]):
        """تنظیم نتایج پرس‌وجو برای تست"""
        self._query_results = results



class MockCache(CachingInterface):
    """پیاده‌سازی Mock برای تست CachingInterface"""

    def __init__(self):
        self._connected = False
        self._data = {}  # {key: (value, expiry_time)}
        self._ttls = {}  # {key: ttl}

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def get(self, key: str) -> Any:
        if key not in self._data:
            return None
        value = self._data.get(key)
        if self._is_expired(key):
            await self.delete(key)
            return None
        return value

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        self._data[key] = value
        if ttl is not None:
            self._ttls[key] = ttl

    async def delete(self, key: str) -> bool:
        if key in self._data:
            del self._data[key]
            self._ttls.pop(key, None)
            return True
        return False

    async def exists(self, key: str) -> bool:
        return key in self._data and not self._is_expired(key)

    async def ttl(self, key: str) -> Optional[int]:
        """دریافت زمان انقضای باقی‌مانده برای یک کلید"""
        if key not in self._data:
            return None
        if key not in self._ttls:
            return None
        if self._is_expired(key):
            await self.delete(key)
            return None
        return self._ttls[key]

    async def scan_keys(self, pattern: str) -> list:
        """جستجوی کلیدها با الگو"""
        import fnmatch
        matched_keys = []
        for key in self._data.keys():
            if fnmatch.fnmatch(key, pattern):
                if not self._is_expired(key):
                    matched_keys.append(key)
        return matched_keys

    def _is_expired(self, key: str) -> bool:
        """بررسی منقضی شدن یک کلید"""
        if key not in self._ttls:
            return False
        return self._ttls[key] <= 0

    def set_ttl(self, key: str, ttl: int) -> None:
        """تنظیم TTL برای یک کلید (برای تست)"""
        if key in self._data:
            self._ttls[key] = ttl


class MockMessaging(MessagingInterface):
    """پیاده‌سازی Mock برای تست MessagingInterface"""

    def __init__(self):
        self._connected = False
        self._subscribers = {}
        self._messages = []

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def publish(self, topic: str, message: Any) -> None:
        self._messages.append((topic, message))
        if topic in self._subscribers:
            await self._subscribers[topic](message)

    async def subscribe(self, topic: str, handler) -> None:
        self._subscribers[topic] = handler

    async def unsubscribe(self, topic: str) -> None:
        self._subscribers.pop(topic, None)


# تست‌های واحد


@pytest.mark.asyncio
async def test_storage_interface():
    """تست رفتار StorageInterface"""
    storage = MockStorage()

    # تست اتصال اولیه
    assert not await storage.is_connected()
    await storage.connect()
    assert await storage.is_connected()

    # تست عملیات جدول
    table_schema = {
        'id': 'INTEGER',
        'name': 'TEXT'
    }
    await storage.create_table('test_table', table_schema)

    # تست تراکنش
    await storage.begin_transaction()
    await storage.execute("INSERT INTO test_table (id, name) VALUES (1, 'test')")
    await storage.commit()

    # تست rollback
    await storage.begin_transaction()
    await storage.execute("INSERT INTO test_table (id, name) VALUES (2, 'test2')")
    await storage.rollback()

    # تست خطای اتصال
    storage.set_available(False)
    with pytest.raises(ConnectionError):
        await storage.connect()

    # تست قطع اتصال
    await storage.disconnect()
    assert not await storage.is_connected()


@pytest.mark.asyncio
async def test_cache_interface():
    """تست رفتار CachingInterface"""
    cache = MockCache()
    await cache.connect()

    # تست ذخیره و بازیابی
    await cache.set("test_key", "test_value", ttl=100)
    assert await cache.exists("test_key")
    assert await cache.get("test_key") == "test_value"
    assert await cache.ttl("test_key") == 100

    # تست scan_keys
    await cache.set("test:1", "value1")
    await cache.set("test:2", "value2")
    await cache.set("other:1", "value3")

    test_keys = await cache.scan_keys("test:*")
    assert len(test_keys) == 2
    assert all(key.startswith("test:") for key in test_keys)

    # تست حذف
    assert await cache.delete("test_key")
    assert not await cache.exists("test_key")
    assert await cache.get("test_key") is None
    assert await cache.ttl("test_key") is None

    # تست کلید منقضی شده
    await cache.set("expired_key", "value", ttl=0)
    assert not await cache.exists("expired_key")
    assert await cache.get("expired_key") is None


@pytest.mark.asyncio
async def test_messaging_interface():
    """تست رفتار MessagingInterface"""

    messaging = MockMessaging()
    await messaging.connect()

    received_messages = []

    async def message_handler(message):
        received_messages.append(message)

    # تست انتشار و دریافت پیام
    await messaging.subscribe("test_topic", message_handler)
    await messaging.publish("test_topic", "test_message")
    assert received_messages == ["test_message"]

    # تست لغو اشتراک
    await messaging.unsubscribe("test_topic")
    await messaging.publish("test_topic", "another_message")
    assert len(received_messages) == 1  # پیام جدید دریافت نشده است