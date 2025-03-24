import psutil
import asyncio
import logging
import zlib
from typing import Optional, List, Any

logging.basicConfig(level=logging.INFO)


class MemoryOptimizer:
    def __init__(self,
                 buffer,
                 max_memory_usage: float = 0.75,
                 critical_memory_threshold: float = 0.9):
        """
        بهینه‌ساز مصرف حافظه برای پردازش‌های جریانی

        :param buffer: نمونه‌ای از SmartBuffer برای مدیریت داده‌ها
        :param max_memory_usage: حداکثر درصد استفاده از حافظه قبل از اجرای پاک‌سازی (پیش‌فرض: 75٪)
        :param critical_memory_threshold: درصد بحرانی استفاده از حافظه که تخلیه اضطراری را اجرا می‌کند (پیش‌فرض: 90٪)
        """
        self.buffer = buffer
        self.max_memory_usage = max_memory_usage
        self.critical_memory_threshold = critical_memory_threshold
        self.monitoring_interval = 2  # بررسی وضعیت حافظه هر ۲ ثانیه
        self.running = True

    async def monitor_memory_usage(self):
        """
        پایش میزان مصرف حافظه و اجرای تخلیه در صورت نیاز
        """
        while self.running:
            memory_usage = self.get_memory_usage()
            logging.info(f"📊 Memory Usage: {memory_usage:.2f}%")

            if memory_usage > self.critical_memory_threshold * 100:
                logging.critical("🚨 CRITICAL MEMORY USAGE! Forcing immediate cleanup!")
                await self.force_cleanup()

            elif memory_usage > self.max_memory_usage * 100:
                logging.warning("⚠️ High memory usage detected! Initiating cleanup...")
                await self.optimize_memory()

            await asyncio.sleep(self.monitoring_interval)

    def get_memory_usage(self) -> float:
        """
        دریافت میزان استفاده از حافظه سیستم (بر حسب درصد)

        :return: درصد استفاده از RAM
        """
        return psutil.virtual_memory().percent

    async def optimize_memory(self):
        """
        کاهش مصرف حافظه از طریق حذف و فشرده‌سازی داده‌ها
        """
        buffer_size = await self.buffer.size()

        if buffer_size > 0:
            # **اولویت‌بندی حذف داده‌ها**
            num_items_to_remove = int(buffer_size * 0.25)  # حذف ۲۵٪ از داده‌های قدیمی‌تر

            for _ in range(num_items_to_remove):
                item = await self.buffer.get()  # حذف قدیمی‌ترین آیتم

                if isinstance(item, bytes):
                    item = self.compress_data(item)  # فشرده‌سازی قبل از حذف

                logging.info(f"🗑️ Removed item from buffer to free memory.")

        else:
            logging.info("✅ Buffer is already empty. No need for optimization.")

    async def force_cleanup(self):
        """
        **پاک‌سازی اضطراری حافظه در صورت رسیدن به حد بحرانی**
        """
        buffer_size = await self.buffer.size()

        if buffer_size > 0:
            logging.critical("🚨 Emergency Cleanup Activated! Removing 50% of buffer data.")
            num_items_to_remove = int(buffer_size * 0.5)  # حذف ۵۰٪ داده‌ها

            for _ in range(num_items_to_remove):
                await self.buffer.get()  # حذف آیتم‌ها

        else:
            logging.critical("❗ Buffer already empty. No more cleanup required.")

    @staticmethod
    def compress_data(data: Any) -> bytes:
        """
        **فشرده‌سازی داده‌ها قبل از حذف برای کاهش مصرف حافظه**

        :param data: داده ورودی
        :return: داده فشرده‌شده
        """
        if isinstance(data, str):
            data = data.encode()
        compressed = zlib.compress(data)
        logging.info(f"📦 Compressed data size: {len(compressed)} bytes")
        return compressed

    async def stop(self):
        """
        توقف فرآیند مانیتورینگ
        """
        self.running = False
        logging.info("⛔ Stopping Memory Optimizer monitoring.")
