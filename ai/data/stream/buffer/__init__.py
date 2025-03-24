import asyncio
import logging

from .smart_buffer import SmartBuffer
from .memory_optimizer import MemoryOptimizer
from .overflow_handler import OverflowHandler

logging.basicConfig(level=logging.INFO)


class BufferManager:
    def __init__(self,
                 max_size: int = 10000,
                 eviction_policy: str = "FIFO",
                 overflow_strategy: str = "drop_oldest",
                 memory_threshold: float = 0.75,
                 critical_memory_threshold: float = 0.9):
        """
        مدیریت یکپارچه‌ی بافر، بهینه‌سازی حافظه، و مدیریت سرریز داده‌ها

        :param max_size: حداکثر اندازه‌ی بافر
        :param eviction_policy: سیاست حذف داده‌ها (FIFO, LIFO, PRIORITY)
        :param overflow_strategy: استراتژی سرریز (drop_oldest, drop_newest, block, compress_oldest)
        :param memory_threshold: حد آستانه‌ی حافظه برای فعال‌سازی تخلیه‌ی داده‌ها (پیش‌فرض: ۷۵٪)
        :param critical_memory_threshold: حد بحرانی حافظه برای پاک‌سازی اضطراری (پیش‌فرض: ۹۰٪)
        """
        self.buffer = SmartBuffer(max_size, eviction_policy, overflow_strategy)
        self.memory_optimizer = MemoryOptimizer(self.buffer, memory_threshold, critical_memory_threshold)
        self.overflow_handler = OverflowHandler(self.buffer, overflow_strategy)

    async def start_monitoring(self):
        """
        شروع مانیتورینگ حافظه و مدیریت خودکار سرریز داده‌ها
        """
        asyncio.create_task(self.memory_optimizer.monitor_memory_usage())
        asyncio.create_task(self.overflow_handler.monitor_overflow())

    async def stop_monitoring(self):
        """
        متوقف کردن مانیتورینگ حافظه و سرریز
        """
        await self.memory_optimizer.stop()
        await self.overflow_handler.stop()

    async def add_data(self, item, priority=None):
        """
        اضافه کردن داده به بافر (با مدیریت خودکار سرریز)
        """
        success = await self.buffer.add(item, priority)
        if not success:
            logging.warning("⚠️ Data insertion failed due to buffer overflow!")
        return success

    async def get_data(self):
        """
        دریافت یک داده از بافر
        """
        return await self.buffer.get()

    async def clear_buffer(self):
        """
        پاک‌سازی کامل بافر
        """
        await self.buffer.clear()

    async def buffer_size(self):
        """
        دریافت تعداد آیتم‌های موجود در بافر
        """
        return await self.buffer.size()


# مقداردهی اولیه‌ی ماژول
buffer_manager = BufferManager()

# راه‌اندازی مانیتورینگ خودکار حافظه
asyncio.create_task(buffer_manager.start_monitoring())

# API ماژول
__all__ = ["buffer_manager", "SmartBuffer", "MemoryOptimizer", "OverflowHandler"]
