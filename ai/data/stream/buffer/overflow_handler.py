import asyncio
import logging
import zlib
from typing import Any, Optional

logging.basicConfig(level=logging.INFO)


class OverflowHandler:
    def __init__(self, buffer, strategy: str = "drop_oldest"):
        """
        مدیریت سرریز داده در بافر جریانی

        :param buffer: نمونه‌ای از SmartBuffer برای مدیریت سرریز
        :param strategy: سیاست مدیریت سرریز (drop_oldest, drop_newest, block, compress_oldest)
        """
        self.buffer = buffer
        self.strategy = strategy.lower()
        self.monitoring_interval = 1  # بررسی وضعیت بافر هر ۱ ثانیه
        self.running = True

    async def monitor_overflow(self):
        """
        پایش میزان اشغال بافر و اجرای استراتژی مدیریت سرریز در صورت نیاز
        """
        while self.running:
            buffer_size = await self.buffer.size()

            if buffer_size >= self.buffer.max_size:
                logging.warning(f"⚠️ Buffer Overflow Detected! Applying strategy: {self.strategy}")
                await self.handle_overflow()

            await asyncio.sleep(self.monitoring_interval)

    async def handle_overflow(self):
        """
        اعمال سیاست مدیریت سرریز داده‌ها
        """
        if self.strategy == "drop_oldest":
            await self._drop_oldest()
        elif self.strategy == "drop_newest":
            await self._drop_newest()
        elif self.strategy == "block":
            await self._block_until_free_space()
        elif self.strategy == "compress_oldest":
            await self._compress_oldest()
        else:
            logging.error(f"❌ Unknown overflow strategy: {self.strategy}")

    async def _drop_oldest(self):
        """
        حذف قدیمی‌ترین داده‌ها از بافر
        """
        removed_item = await self.buffer.get()
        logging.info(f"🗑️ Dropped oldest item: {removed_item}")

    async def _drop_newest(self):
        """
        حذف جدیدترین داده از بافر
        """
        if len(self.buffer.buffer) > 0:
            removed_item = self.buffer.buffer.pop()
            logging.info(f"🗑️ Dropped newest item: {removed_item}")

    async def _block_until_free_space(self):
        """
        متوقف کردن ورود داده‌های جدید تا زمانی که فضای خالی در بافر ایجاد شود
        """
        logging.warning("🚧 Buffer is full! Blocking new data entry until space is available...")
        while await self.buffer.size() >= self.buffer.max_size:
            await asyncio.sleep(0.5)  # انتظار برای آزاد شدن فضا

    async def _compress_oldest(self):
        """
        فشرده‌سازی قدیمی‌ترین داده‌های بافر به جای حذف مستقیم
        """
        item = await self.buffer.get()
        if item:
            compressed_item = self.compress_data(item)
            logging.info(f"📦 Compressed oldest item. New size: {len(compressed_item)} bytes")
            await self.buffer.add(compressed_item)  # داده فشرده را مجدداً به بافر برمی‌گردانیم

    @staticmethod
    def compress_data(data: Any) -> bytes:
        """
        **فشرده‌سازی داده‌ها برای کاهش مصرف حافظه**

        :param data: داده‌ای که باید فشرده شود
        :return: داده فشرده‌شده
        """
        if isinstance(data, str):
            data = data.encode()
        compressed = zlib.compress(data)
        return compressed

    async def stop(self):
        """
        توقف فرآیند پایش سرریز
        """
        self.running = False
        logging.info("⛔ Stopping OverflowHandler monitoring.")
