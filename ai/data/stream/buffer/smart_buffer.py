import asyncio
import time
import logging
from collections import deque
from typing import Any, Optional, List

logging.basicConfig(level=logging.INFO)


class SmartBuffer:
    def __init__(self,
                 max_size: int = 10000,
                 eviction_policy: str = "FIFO",
                 overflow_strategy: str = "drop_oldest"):
        """
        بافر هوشمند برای ذخیره‌سازی داده‌های جریانی با قابلیت کنترل حافظه

        :param max_size: حداکثر تعداد آیتم‌هایی که بافر می‌تواند نگه دارد
        :param eviction_policy: سیاست حذف داده‌ها (FIFO, LIFO, PRIORITY)
        :param overflow_strategy: استراتژی مدیریت سرریز (drop_oldest, drop_newest, block)
        """
        self.max_size = max_size
        self.eviction_policy = eviction_policy.upper()
        self.overflow_strategy = overflow_strategy.lower()
        self.buffer = deque()
        self.lock = asyncio.Lock()

    async def add(self, item: Any, priority: Optional[int] = None) -> bool:
        """
        اضافه کردن یک آیتم به بافر. در صورت پر شدن، بر اساس استراتژی سرریز مدیریت می‌شود.

        :param item: داده‌ای که باید به بافر اضافه شود
        :param priority: مقدار اولویت برای سیاست PRIORITY
        :return: True اگر آیتم با موفقیت اضافه شود، False در صورت حذف به دلیل سرریز
        """
        async with self.lock:
            if len(self.buffer) >= self.max_size:
                logging.warning(f"⚠️  Buffer is full! Applying overflow strategy: {self.overflow_strategy}")
                if self.overflow_strategy == "drop_oldest":
                    self.buffer.popleft()  # حذف قدیمی‌ترین آیتم
                elif self.overflow_strategy == "drop_newest":
                    return False  # آیتم جدید را حذف می‌کند
                elif self.overflow_strategy == "block":
                    while len(self.buffer) >= self.max_size:
                        await asyncio.sleep(0.01)  # منتظر می‌ماند تا فضا خالی شود

            if self.eviction_policy == "LIFO":
                self.buffer.appendleft(item)  # اضافه به اول لیست (LIFO)
            elif self.eviction_policy == "PRIORITY":
                self._insert_with_priority(item, priority)
            else:  # پیش‌فرض FIFO
                self.buffer.append(item)

            return True

    async def get(self) -> Optional[Any]:
        """
        دریافت یک آیتم از بافر بر اساس سیاست ذخیره‌سازی

        :return: آیتمی از بافر (یا None اگر بافر خالی باشد)
        """
        async with self.lock:
            if self.buffer:
                if self.eviction_policy == "LIFO":
                    return self.buffer.popleft()  # دریافت از ابتدا (LIFO)
                else:
                    return self.buffer.pop()  # دریافت از انتها (FIFO)
            return None

    def _insert_with_priority(self, item: Any, priority: Optional[int]):
        """
        وارد کردن داده به صورت **اولویت‌بندی‌شده**
        - هرچه مقدار priority کمتر باشد، اولویت بیشتر است.
        """
        if priority is None:
            priority = 100  # اولویت پیش‌فرض پایین

        for idx, (existing_priority, _) in enumerate(self.buffer):
            if priority < existing_priority:
                self.buffer.insert(idx, (priority, item))
                return

        self.buffer.append((priority, item))  # اگر هیچ جایی نبود، به انتهای صف اضافه می‌شود

    async def clear(self):
        """
        پاکسازی کل بافر
        """
        async with self.lock:
            self.buffer.clear()
            logging.info("✅ Buffer has been cleared.")

    async def size(self) -> int:
        """
        تعداد آیتم‌های موجود در بافر را برمی‌گرداند.

        :return: تعداد آیتم‌ها
        """
        async with self.lock:
            return len(self.buffer)

    async def peek(self) -> Optional[Any]:
        """
        دریافت آیتم بعدی در صف **بدون حذف کردن آن**
        :return: آیتم بعدی یا None اگر بافر خالی باشد
        """
        async with self.lock:
            if self.buffer:
                return self.buffer[-1]  # آخرین آیتم را بدون حذف برمی‌گرداند
            return None

    async def is_empty(self) -> bool:
        """
        بررسی خالی بودن بافر
        :return: True اگر بافر خالی باشد، در غیر اینصورت False
        """
        async with self.lock:
            return len(self.buffer) == 0
